"""Server-staged step data plane: plan, stage-in, stage-out, key map (ADR-074 D-5)."""

from __future__ import annotations

import io
import tarfile
import threading
from types import SimpleNamespace

import pytest
from s3transfer.upload import UploadNonSeekableInputManager, UploadSeekableInputManager
from umrs.pipeline_sdk.staged_s3 import StagedS3Client
from umrs.pipeline_sdk.step import StepContext, StepResult
from umrs.server import step_staging
from umrs.server.step_executor import StepExecutionError
from umrs.server.step_staging import (
    STAGE_REQUEST_METADATA_KEY,
    OutputUploadError,
    StageCaps,
    StageStreamError,
    build_stage_plan,
    collect_out,
    register_bucket_registry,
    rewrite_output_keys,
    stage_in,
    stage_request,
    verify_stage_summary,
)
from umrs.server.workflow_execution_registry import ExecutionRecord
from umrs.server.workflow_runner import _InMemoryS3Client
from umrs.shared.buckets import BucketsCfg
from umrs.workflow.pg_registry import DurableExecutionRecord

IN, OUT = "alice.p.a", "alice.p.a-output"


@pytest.fixture(autouse=True)
def _no_registry():
    register_bucket_registry(None)
    yield
    register_bucket_registry(None)


class _Body:
    """StreamingBody stand-in: records every read size."""

    def __init__(self, data: bytes, reads: list) -> None:
        self._buf, self._reads = io.BytesIO(data), reads
        self.closed = False

    def read(self, n=None):
        assert n is not None, "the stager must never read an S3 body whole"
        self._reads.append(n)
        return self._buf.read(n)

    def close(self):
        self.closed = True


class _FakeS3:
    def __init__(self, objects=None, page_size=1000):
        self.objects = dict(objects or {})
        self.page_size = page_size
        self.reads: list[int] = []
        self.calls: list[tuple] = []
        self.bodies: list[_Body] = []
        self.uploads: dict[str, bytes] = {}

    def head_object(self, *, Bucket, Key):
        self.calls.append(("head", Bucket, Key))
        data = self.objects[(Bucket, Key)]
        return {"ContentLength": len(data), "ETag": '"t"', "ContentType": "video/mp4"}

    def get_object(self, *, Bucket, Key):
        self.calls.append(("get", Bucket, Key))
        body = _Body(self.objects[(Bucket, Key)], self.reads)
        self.bodies.append(body)
        return {"Body": body, "ContentLength": len(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        self.calls.append(("list", Bucket, Prefix))
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        start = keys.index(ContinuationToken) + 1 if ContinuationToken else 0
        page = keys[start : start + self.page_size]
        more = start + self.page_size < len(keys)
        return {
            "Contents": [
                {"Key": k, "Size": len(self.objects[(Bucket, k)]), "ETag": '"l"'} for k in page
            ],
            "IsTruncated": more,
            **({"NextContinuationToken": page[-1]} if more else {}),
        }

    def upload_fileobj(self, fileobj, bucket, key):
        # what boto3's transfer manager would pick for this stream
        assert UploadNonSeekableInputManager.is_compatible(fileobj)
        assert not UploadSeekableInputManager.is_compatible(fileobj)
        chunks = []
        while chunk := fileobj.read(3):
            chunks.append(chunk)
        self.uploads[f"{bucket}/{key}"] = b"".join(chunks)


def _ctx(input_keys=(), metadata=None, bucket=IN, s3=None, run_id="wf-0123456789ab"):
    return StepContext(
        s3=s3,
        input_bucket=bucket,
        input_keys=list(input_keys),
        output_bucket=bucket + "-output",
        parent_output_bucket=bucket + "-output" if input_keys else "",
        metadata=dict(metadata or {}),
        run_id=run_id,
    )


def _req(stage_inputs=None, params=None, owner="alice", step="seg"):
    spec = {"name": step, "stage_inputs": stage_inputs or {}}
    return stage_request(spec, params or {}, owner)


# ── plan: scope ─────────────────────────────────────────────────────────────


def test_stage_request_carries_only_the_named_params():
    req = _req({"params": ["video"]}, {"video": "s3://alice.p.a/v.mp4", "api_key": "sk"})
    assert req == {
        "owner": "alice",
        "step": "seg",
        "stage_inputs": {"params": ["video"]},
        "params": {"video": "s3://alice.p.a/v.mp4"},
    }


def test_plan_stages_parent_outputs_and_declared_inputs():
    s3 = _FakeS3(
        {
            (OUT, "runs/wf-1/prev/a.npy"): b"aa",
            (IN, "videos/v.mp4"): b"vvv",
            (IN, "frames/1.png"): b"1",
            (IN, "frames/2.png"): b"22",
            (IN, "frames/"): b"",  # folder marker
        }
    )
    ctx = _ctx(
        ["runs/wf-1/prev/a.npy"],
        {
            STAGE_REQUEST_METADATA_KEY: _req(
                {"params": ["video"], "prefixes": ["frames/"]}, {"video": "videos/v.mp4"}
            )
        },
    )
    plan = build_stage_plan(ctx, s3)
    got = {(o.bucket, o.key): (o.path, o.size) for o in plan.objects}
    assert got == {
        (OUT, "runs/wf-1/prev/a.npy"): (f"{OUT}/runs/wf-1/prev/a.npy", 2),
        (IN, "videos/v.mp4"): (f"{IN}/videos/v.mp4", 3),
        (IN, "frames/1.png"): (f"{IN}/frames/1.png", 1),
        (IN, "frames/2.png"): (f"{IN}/frames/2.png", 2),
    }
    manifest = plan.manifest()
    assert manifest["prefixes"] == [f"{IN}/frames/"]
    assert manifest["output_bucket"] == OUT
    assert manifest["objects"][f"{IN}/videos/v.mp4"]["content_type"] == "video/mp4"
    assert plan.step == "seg" and plan.total_bytes == 8


@pytest.mark.parametrize(
    "bad",
    [
        "s3://bob.p.a/v.mp4",  # another owner
        "s3://alice.bob.p.a/v.mp4",  # usernames may carry dots: not alice's
        "s3://alice-x.p.a/v.mp4",
        "s3://wf-runner/v.mp4",  # another run's in-memory store
    ],
)
def test_plan_refuses_a_bucket_the_owner_does_not_own(bad):
    s3 = _FakeS3()
    ctx = _ctx(metadata={STAGE_REQUEST_METADATA_KEY: _req({"params": ["v"]}, {"v": bad})})
    with pytest.raises(StepExecutionError, match="does not belong|in-memory store"):
        build_stage_plan(ctx, s3)
    assert s3.calls == []  # refused before any S3 call


def test_plan_refuses_a_prefix_in_another_owners_bucket_before_listing():
    s3 = _FakeS3({("bob.p.a", "x/1"): b"1"})
    ctx = _ctx(metadata={STAGE_REQUEST_METADATA_KEY: _req({"prefixes": ["s3://bob.p.a/x/"]})})
    with pytest.raises(StepExecutionError, match="does not belong"):
        build_stage_plan(ctx, s3)
    assert s3.calls == []


def test_plan_refuses_when_the_run_bucket_is_not_the_owners():
    ctx = _ctx(bucket="bob.p.a", metadata={STAGE_REQUEST_METADATA_KEY: _req()})
    with pytest.raises(StepExecutionError, match="does not belong"):
        build_stage_plan(ctx, _FakeS3())


def test_a_bucket_named_like_the_in_memory_store_gets_no_bypass():
    # only the exact synthetic names are the run's in-memory store
    ctx = _ctx(bucket="wf-runner.p.a", metadata={STAGE_REQUEST_METADATA_KEY: _req(owner="mallory")})
    with pytest.raises(StepExecutionError, match="does not belong"):
        build_stage_plan(ctx, _FakeS3())


def test_plan_without_an_owner_is_refused():
    with pytest.raises(StepExecutionError, match="run owner"):
        build_stage_plan(_ctx(), _FakeS3())
    with pytest.raises(StepExecutionError, match="run owner"):
        build_stage_plan(_ctx(metadata={STAGE_REQUEST_METADATA_KEY: _req(owner="")}), _FakeS3())


def _registry(**owners):
    cfg = BucketsCfg()
    for bucket, owner in owners.items():
        cfg.store(bucket, SimpleNamespace(s3_username=owner, output_path_to_bucket=None))
    return cfg


def test_the_registry_tells_a_b_from_a_dash_b():
    ctx = _ctx(bucket="a-b.p.a", metadata={STAGE_REQUEST_METADATA_KEY: _req(owner="a-b")})
    register_bucket_registry(_registry(**{"a-b.p.a": "a_b"}))
    with pytest.raises(StepExecutionError, match="registry records another owner"):
        build_stage_plan(ctx, _FakeS3())
    register_bucket_registry(_registry(**{"a-b.p.a": "a-b"}))
    assert build_stage_plan(ctx, _FakeS3()).output_bucket == "a-b.p.a-output"


def test_an_unprovisioned_bucket_is_refused_once_the_registry_is_loaded():
    register_bucket_registry(_registry(**{IN: "alice"}))
    ctx = _ctx(
        metadata={STAGE_REQUEST_METADATA_KEY: _req({"params": ["v"]}, {"v": "s3://alice.q.z/k"})}
    )
    with pytest.raises(StepExecutionError, match="not a provisioned bucket"):
        build_stage_plan(ctx, _FakeS3({("alice.q.z", "k"): b"k"}))


def test_the_runs_own_in_memory_store_is_staged():
    record = ExecutionRecord(execution_id="wf-mem", workflow_name="w", user_id="alice")
    s3 = _InMemoryS3Client(record, state_lock=threading.Lock())
    s3.put_object(Bucket="wf-runner", Key="in/a.txt", Body=b"hello")
    ctx = _ctx(
        bucket="wf-runner",
        s3=s3,
        metadata={STAGE_REQUEST_METADATA_KEY: _req({"params": ["f"]}, {"f": "in/a.txt"})},
    )
    plan = build_stage_plan(ctx)
    buf = io.BytesIO()
    stage_in(plan, s3, buf)
    assert _stage_in_members(buf)["in/wf-runner/in/a.txt"] == b"hello"


def test_prefixes_need_a_listable_bucket():
    record = ExecutionRecord(execution_id="wf-mem", workflow_name="w", user_id="alice")
    s3 = _InMemoryS3Client(record, state_lock=threading.Lock())
    ctx = _ctx(
        bucket="wf-runner",
        s3=s3,
        metadata={STAGE_REQUEST_METADATA_KEY: _req({"prefixes": ["in/"]})},
    )
    with pytest.raises(StepExecutionError, match="needs a real bucket to list"):
        build_stage_plan(ctx)


def test_a_missing_input_is_refused_by_name():
    ctx = _ctx(metadata={STAGE_REQUEST_METADATA_KEY: _req({"params": ["v"]}, {"v": "nope"})})
    with pytest.raises(StepExecutionError, match=r"s3://alice\.p\.a/nope is not readable"):
        build_stage_plan(ctx, _FakeS3())


def test_non_string_param_values_are_refused():
    ctx = _ctx(metadata={STAGE_REQUEST_METADATA_KEY: _req({"params": ["v"]}, {"v": {"k": 1}})})
    with pytest.raises(StepExecutionError, match="param v"):
        build_stage_plan(ctx, _FakeS3())


# ── plan: caps and placement ────────────────────────────────────────────────


def _plan_with(objects, caps, stage_inputs=None, params=None):
    ctx = _ctx(metadata={STAGE_REQUEST_METADATA_KEY: _req(stage_inputs, params)})
    return build_stage_plan(ctx, _FakeS3(objects, page_size=2), caps=caps)


def test_per_object_per_step_and_count_caps():
    objects = {(IN, f"d/{i}"): b"x" * 4 for i in range(3)}
    si = {"prefixes": ["d/"]}
    assert len(_plan_with(objects, StageCaps(), si).objects) == 3  # paged listing
    with pytest.raises(StepExecutionError, match="per-object cap"):
        _plan_with(objects, StageCaps(max_object_bytes=3), si)
    with pytest.raises(StepExecutionError, match="per-step cap"):
        _plan_with(objects, StageCaps(max_in_bytes=11), si)
    with pytest.raises(StepExecutionError, match="more than 2 objects"):
        _plan_with(objects, StageCaps(max_in_objects=2), si)
    many = {"params": ["v"]}
    with pytest.raises(StepExecutionError, match="3 objects"):
        _plan_with(objects, StageCaps(max_in_objects=2), many, {"v": ["d/0", "d/1", "d/2"]})


def _is_safe_rel(path):
    return ".." not in path.split("/") and not path.startswith("/")


def test_awkward_keys_get_collision_free_paths():
    keys = ["a", "a/b", "x//y", "../up", ".hidden", "sp ace", "ok/_SUCCESS"]
    objects = {(IN, k): b"1" for k in keys}
    plan = _plan_with(objects, StageCaps(), {"params": ["v"]}, {"v": keys})
    paths = {o.key: o.path for o in plan.objects}
    # every other key ("a/b" collides with the file "a") lands hashed
    natural = {k: p for k, p in paths.items() if not p.startswith(f"{IN}/.umrs-hashed/")}
    assert natural == {"a": f"{IN}/a", "ok/_SUCCESS": f"{IN}/ok/_SUCCESS"}
    assert len(set(paths.values())) == len(keys)
    assert all(map(_is_safe_rel, paths.values()))


# ── stage-in stream ─────────────────────────────────────────────────────────


class _RecordingSink(io.BytesIO):
    """Stage-in sink that records every write size."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[int] = []

    def write(self, data):
        self.writes.append(len(data))
        return super().write(data)


def _stage_in_members(buf):
    """Read a stage-in stream back; every member must be a 0600 regular file."""
    buf.seek(0)
    members = {}
    with tarfile.open(fileobj=buf, mode="r|") as tar:
        for m in tar:
            assert m.isreg() and m.mode == 0o600
            members[m.name] = tar.extractfile(m).read()
    return members


def _source_tree(tmp_path):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "step.py").write_text("X = 1\n")
    return str(src)


def test_stage_in_streams_members_without_buffering_objects(tmp_path):
    big = bytes(range(256)) * (3 * 4096 + 7)  # ~3 MiB
    s3 = _FakeS3({(IN, "v.bin"): big, (OUT, "runs/r/p/k"): b"k"})
    ctx = _ctx(
        ["runs/r/p/k"],
        {STAGE_REQUEST_METADATA_KEY: _req({"params": ["v"]}, {"v": "v.bin"})},
    )
    plan = build_stage_plan(ctx, s3)
    sink = _RecordingSink()
    sent = stage_in(plan, s3, sink, source_dir=_source_tree(tmp_path))
    assert max(s3.reads) <= 1024 * 1024  # chunked body reads
    assert max(sink.writes) <= 64 * 1024  # streamed out in records
    assert all(body.closed for body in s3.bodies)
    members = _stage_in_members(sink)
    assert members == {
        "in/manifest.json": plan.manifest_bytes(),
        f"in/{IN}/v.bin": big,
        f"in/{OUT}/runs/r/p/k": b"k",
        "src/pkg/step.py": b"X = 1\n",
    }
    assert sent == {"files": 4, "bytes": sum(len(v) for v in members.values())}


def test_stage_in_refuses_an_object_that_changed_size():
    s3 = _FakeS3({(IN, "v"): b"abc"})
    ctx = _ctx(metadata={STAGE_REQUEST_METADATA_KEY: _req({"params": ["v"]}, {"v": "v"})})
    plan = build_stage_plan(ctx, s3)
    s3.objects[(IN, "v")] = b"abcd"
    with pytest.raises(StepExecutionError, match="changed size"):
        stage_in(plan, s3, io.BytesIO())


def test_stage_in_refuses_a_body_shorter_than_its_length():
    s3 = _FakeS3({(IN, "v"): b"abc"})
    ctx = _ctx(metadata={STAGE_REQUEST_METADATA_KEY: _req({"params": ["v"]}, {"v": "v"})})
    plan = build_stage_plan(ctx, s3)
    real_get = s3.get_object

    def short_get(**kw):
        got = real_get(**kw)
        got["Body"] = _Body(b"ab", [])
        return got

    s3.get_object = short_get
    with pytest.raises(StepExecutionError, match="ended early"):
        stage_in(plan, s3, io.BytesIO())


def test_stage_in_refuses_a_symlink_in_the_source_tree(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "leak").symlink_to("/etc/passwd")
    plan = build_stage_plan(_ctx(metadata={STAGE_REQUEST_METADATA_KEY: _req()}), _FakeS3())
    with pytest.raises(StepExecutionError, match="links refused"):
        stage_in(plan, _FakeS3(), io.BytesIO(), source_dir=str(src))


def test_stage_in_reports_a_broken_stream():
    class _Broken(io.RawIOBase):
        def write(self, _data):
            raise BrokenPipeError("ssh went away")

    plan = build_stage_plan(_ctx(metadata={STAGE_REQUEST_METADATA_KEY: _req()}), _FakeS3())
    with pytest.raises(StageStreamError, match="stage-in stream broke"):
        stage_in(plan, _FakeS3(), _Broken())


def test_verify_stage_summary():
    caps = StageCaps(min_free_bytes=100)
    verify_stage_summary(
        {"files": 2, "bytes": 5}, {"files": 2, "bytes": 5, "free_bytes": 100}, caps
    )
    with pytest.raises(StepExecutionError, match="stage-in wrote 1 files"):
        verify_stage_summary({"files": 2, "bytes": 5}, {"files": 1, "bytes": 5}, caps)
    with pytest.raises(StepExecutionError, match="bytes free"):
        verify_stage_summary(
            {"files": 2, "bytes": 5}, {"files": 2, "bytes": 5, "free_bytes": 99}, caps
        )


# ── stage-out stream ────────────────────────────────────────────────────────


def _tar(*members, fmt=tarfile.PAX_FORMAT) -> io.BytesIO:
    """members: (TarInfo, data-or-None)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as tar:
        for info, data in members:
            if data is not None:
                info.size = len(data)
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    buf.seek(0)
    return buf


def _file(name, data=b"x"):
    return tarfile.TarInfo(name), data


def _special(name, kind, linkname=""):
    info = tarfile.TarInfo(name)
    info.type, info.linkname = kind, linkname
    return info, None


def test_collect_out_uploads_regular_files_to_the_run_prefix():
    s3 = _FakeS3()
    dirinfo = tarfile.TarInfo(".")
    dirinfo.type = tarfile.DIRTYPE
    stream = _tar(
        (dirinfo, None), _file(f"./{OUT}/res/a.json", b"{}"), _file(f"{OUT}/b.bin", b"bbbbbbb")
    )
    mapping = collect_out(stream, s3, OUT, "wf-0123456789ab", "seg")
    assert mapping == {
        "res/a.json": "runs/wf-0123456789ab/seg/res/a.json",
        "b.bin": "runs/wf-0123456789ab/seg/b.bin",
    }
    assert s3.uploads[f"{OUT}/runs/wf-0123456789ab/seg/b.bin"] == b"bbbbbbb"


@pytest.mark.parametrize(
    ("member", "why"),
    [
        (_special(f"{OUT}/l", tarfile.SYMTYPE, "/home/u/.ssh/id_rsa"), "a symlink"),
        (_special(f"{OUT}/h", tarfile.LNKTYPE, f"{OUT}/x"), "a hard link"),
        (_special(f"{OUT}/c", tarfile.CHRTYPE), "a character device"),
        (_special(f"{OUT}/b", tarfile.BLKTYPE), "a block device"),
        (_special(f"{OUT}/f", tarfile.FIFOTYPE), "a FIFO"),
        (_file(f"{OUT}/../../etc/cron.d/x"), r"'\.\.' segment"),
        (_file(f"/{OUT}/abs"), "absolute name"),
        (_file("loose.txt"), "not under the output bucket"),
        (_file(f"{IN}/k"), "not under the output bucket"),
        (_file(f"{OUT}/bad\x01name"), "control characters"),
    ],
)
def test_collect_out_refuses_hostile_members(member, why):
    s3 = _FakeS3()
    with pytest.raises(StepExecutionError, match=why):
        collect_out(_tar(member), s3, OUT, "wf-1", "seg")
    assert s3.uploads == {}


def test_collect_out_refuses_a_dotdot_directory():
    info = tarfile.TarInfo("../")
    info.type = tarfile.DIRTYPE
    with pytest.raises(StepExecutionError, match=r"'\.\.' segment"):
        collect_out(_tar((info, None)), _FakeS3(), OUT, "wf-1", "seg")


def test_collect_out_refuses_sparse_members():
    info = tarfile.TarInfo(f"{OUT}/s")
    info.type = tarfile.GNUTYPE_SPARSE
    with pytest.raises(StepExecutionError, match="sparse"):
        collect_out(_tar((info, None), fmt=tarfile.GNU_FORMAT), _FakeS3(), OUT, "wf-1", "seg")
    pax = tarfile.TarInfo(f"{OUT}/s")
    pax.pax_headers = {"GNU.sparse.map": "0,1", "GNU.sparse.size": "1"}
    with pytest.raises(StepExecutionError, match="not a regular file"):
        collect_out(_tar((pax, b"x")), _FakeS3(), OUT, "wf-1", "seg")


def test_collect_out_refuses_oversized_tar_metadata_before_buffering_it():
    long_name = tarfile.TarInfo(f"{OUT}/" + "n/" * 40_000 + "f")
    with pytest.raises(StepExecutionError, match="metadata header"):
        collect_out(_tar((long_name, b"x"), fmt=tarfile.GNU_FORMAT), _FakeS3(), OUT, "wf-1", "seg")
    pax = tarfile.TarInfo(f"{OUT}/f")
    pax.pax_headers = {"comment": "c" * 100_000}
    with pytest.raises(StepExecutionError, match="metadata header"):
        collect_out(_tar((pax, b"x")), _FakeS3(), OUT, "wf-1", "seg")


def test_collect_out_caps():
    two = _tar(_file(f"{OUT}/a", b"1234"), _file(f"{OUT}/b", b"1234"))
    with pytest.raises(StepExecutionError, match="more than 1 files"):
        collect_out(two, _FakeS3(), OUT, "wf-1", "seg", StageCaps(max_out_files=1))
    two.seek(0)
    with pytest.raises(StepExecutionError, match="cap 3"):
        collect_out(two, _FakeS3(), OUT, "wf-1", "seg", StageCaps(max_out_file_bytes=3))
    two.seek(0)
    with pytest.raises(StepExecutionError, match="more than 6 bytes"):
        collect_out(two, _FakeS3(), OUT, "wf-1", "seg", StageCaps(max_out_bytes=6))
    dup = _tar(_file(f"{OUT}/a"), _file(f"./{OUT}//a"))
    with pytest.raises(StepExecutionError, match="appears twice"):
        collect_out(dup, _FakeS3(), OUT, "wf-1", "seg")


def test_collect_out_refuses_garbage():
    with pytest.raises(StepExecutionError, match="not a valid tar"):
        collect_out(io.BytesIO(b"\x01" * 1024), _FakeS3(), OUT, "wf-1", "seg")


def _mem_store(record_cls=ExecutionRecord):
    record = record_cls(execution_id="wf-mem", workflow_name="w", user_id="alice")
    return record, _InMemoryS3Client(record, state_lock=threading.Lock())


def test_collect_out_into_the_in_memory_store_is_capped():
    record, s3 = _mem_store()
    stream = _tar(_file("wf-runner-output/r.txt", b"result"))
    mapping = collect_out(stream, s3, "wf-runner-output", "wf-mem", "seg")
    assert record.results["blobs"]["wf-runner-output/runs/wf-mem/seg/r.txt"] == b"result"
    assert mapping == {"r.txt": "runs/wf-mem/seg/r.txt"}
    assert record.artifacts == ["wf-runner-output/runs/wf-mem/seg/r.txt"]
    _record, fresh = _mem_store()
    stream.seek(0)
    with pytest.raises(StepExecutionError, match="more than 5 bytes.*give the run a bucket"):
        collect_out(
            stream, fresh, "wf-runner-output", "wf-mem", "seg", StageCaps(max_buffered_bytes=5)
        )


def test_the_in_memory_budget_is_per_run_not_per_step():
    record, s3 = _mem_store()
    caps = StageCaps(max_buffered_bytes=10)
    s3.put_object(Bucket="wf-runner-output", Key="platform.json", Body=b"123")
    collect_out(_tar(_file("wf-runner-output/a", b"1234")), s3, "wf-runner-output", "wf-mem",
                "s1", caps)  # fmt: skip
    with pytest.raises(StepExecutionError, match="more than 3 bytes"):
        collect_out(_tar(_file("wf-runner-output/b", b"1234")), s3, "wf-runner-output",
                    "wf-mem", "s2", caps)  # fmt: skip
    assert set(record.results["blobs"]) == {
        "wf-runner-output/platform.json",
        "wf-runner-output/runs/wf-mem/s1/a",
    }


def test_a_buffered_step_adds_few_files():
    _record, s3 = _mem_store()
    three = _tar(*(_file(f"wf-runner-output/{i}", b"x") for i in range(3)))
    with pytest.raises(StepExecutionError, match="more than 2 files.*bucket param"):
        collect_out(three, s3, "wf-runner-output", "wf-mem", "s1", StageCaps(max_buffered_files=2))
    assert _record.results["blobs"] == {}  # nothing stored from a refused step


def test_a_buffered_collection_persists_the_durable_record_once():
    record, s3 = _mem_store(DurableExecutionRecord)
    persisted = []
    record._persist_hook = lambda rec: persisted.append(len(rec.results["blobs"]))
    files = [_file(f"wf-runner-output/f{i}", b"data") for i in range(20)]
    mapping = collect_out(_tar(*files), s3, "wf-runner-output", "wf-mem", "s1")
    assert len(mapping) == 20
    assert persisted == [20]  # one snapshot for the batch, not one per file
    assert len(record.artifacts) == 20


def test_the_store_rechecks_the_budget_a_parallel_step_claimed():
    record, s3 = _mem_store()
    stream = _tar(_file("wf-runner-output/a", b"12345678"))
    real = s3.stored_bytes
    s3.stored_bytes = lambda: 0  # the budget looked free when the step began
    s3.put_object(Bucket="wf-runner-output", Key="other", Body=b"123456")
    with pytest.raises(StepExecutionError, match="would hold 14 bytes; its budget is 10"):
        collect_out(stream, s3, "wf-runner-output", "wf-mem", "s1",
                    StageCaps(max_buffered_bytes=10))  # fmt: skip
    assert real() == 6 and "wf-runner-output/runs/wf-mem/s1/a" not in record.results["blobs"]


def test_an_upload_failure_is_an_output_upload_error():
    class _Down(_FakeS3):
        def upload_fileobj(self, *_args):
            raise ConnectionError("seaweedfs down")

    with pytest.raises(OutputUploadError, match="seaweedfs down"):
        collect_out(_tar(_file(f"{OUT}/a", b"1")), _Down(), OUT, "wf-1", "seg")


def test_a_broken_stage_out_stream_is_a_stream_error():
    with pytest.raises(StageStreamError):
        collect_out(io.BytesIO(b""), _FakeS3(), OUT, "wf-1", "seg")


def test_a_prefix_that_cannot_be_listed_is_refused_by_name():
    class _Denied(_FakeS3):
        def list_objects_v2(self, **_kw):
            raise RuntimeError("An error occurred (NoSuchBucket) when calling ListObjectsV2")

    ctx = _ctx(metadata={STAGE_REQUEST_METADATA_KEY: _req({"prefixes": [f"s3://{IN}/raw/"]})})
    with pytest.raises(StepExecutionError, match="not listable: .*NoSuchBucket") as info:
        build_stage_plan(ctx, _Denied())
    assert info.value.phase == "setup"


def test_a_body_that_fails_mid_stream_is_refused_by_name():
    class _Flaky(_FakeS3):
        def get_object(self, *, Bucket, Key):
            class _Body:
                def read(self, n=None):
                    if n == 0:
                        return b""
                    raise RuntimeError("ResponseStreamingError: connection reset")

            return {"Body": _Body(), "ContentLength": len(self.objects[(Bucket, Key)])}

    s3 = _Flaky({(IN, "a.bin"): b"abc"})
    ctx = _ctx(metadata={STAGE_REQUEST_METADATA_KEY: _req({"params": ["x"]}, {"x": "a.bin"})})
    plan = build_stage_plan(ctx, s3)
    with pytest.raises(StepExecutionError, match="could not be read: .*connection reset"):
        stage_in(plan, s3, io.BytesIO())


def test_physical_prefix_keeps_odd_step_names_distinct():
    a = step_staging.physical_prefix("wf-1", "seg a")
    b = step_staging.physical_prefix("wf-1", "seg_a")
    assert a != b and a.startswith("runs/wf-1/seg_a-")
    assert step_staging.physical_prefix("wf-1", "..") != "runs/wf-1/../"


# ── key map ─────────────────────────────────────────────────────────────────


def test_output_keys_are_rewritten_to_physical_keys():
    mapping = {"res/a.json": "runs/wf-1/seg/res/a.json"}
    result = StepResult(
        output_keys=["res/a.json", "/res//a.json", f"s3://{OUT}/res/a.json"], extra={"n": 1}
    )
    rewritten = rewrite_output_keys(result, mapping, OUT)
    assert rewritten.output_keys == [
        "runs/wf-1/seg/res/a.json",
        "runs/wf-1/seg/res/a.json",
        f"s3://{OUT}/runs/wf-1/seg/res/a.json",
    ]
    assert rewritten.extra == {"n": 1}


@pytest.mark.parametrize(
    "bad",
    ["never/written.json", f"s3://{IN}/res/a.json", "../res/a.json", 7, ""],
)
def test_output_keys_that_were_not_collected_are_refused(bad):
    with pytest.raises(StepExecutionError, match="output key"):
        rewrite_output_keys(StepResult(output_keys=[bad]), {"res/a.json": "p"}, OUT)


# ── end to end: plan → stage-in → shim → stage-out → key map → child ────────


def _helper_extract(stream: io.BytesIO, workdir) -> None:
    """What the login-node helper's stage-in does, for the round trip."""
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            dest = workdir / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(member).read())


def _helper_collect(out_dir) -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(out_dir, arcname=".")
    buf.seek(0)
    return buf


def test_round_trip_parent_to_child(tmp_path):
    s3 = _FakeS3({(IN, "videos/v.mp4"): b"video-bytes"})
    ctx = _ctx(
        metadata={
            STAGE_REQUEST_METADATA_KEY: _req(
                {"params": ["video"]}, {"video": "s3://alice.p.a/videos/v.mp4"}, step="parent"
            )
        }
    )
    plan = build_stage_plan(ctx, s3)
    wd = tmp_path / "wd"
    (wd / "out").mkdir(parents=True)
    wire = io.BytesIO()
    stage_in(plan, s3, wire)
    wire.seek(0)
    _helper_extract(wire, wd)

    shim = StagedS3Client(str(wd / "in"), str(wd / "out"))
    step_ctx = StepContext(s3=shim, input_bucket=IN, input_keys=[], output_bucket=OUT)
    data = step_ctx.download_bytes("videos/v.mp4")
    step_ctx.upload_bytes("frames/f0.bin", data.upper())
    reply = StepResult(output_keys=["frames/f0.bin"])

    mapping = collect_out(_helper_collect(wd / "out"), s3, OUT, ctx.run_id, "parent")
    recorded = rewrite_output_keys(reply, mapping, OUT)
    physical = recorded.output_keys[0]
    assert physical == f"runs/{ctx.run_id}/parent/frames/f0.bin"
    assert s3.uploads[f"{OUT}/{physical}"] == b"VIDEO-BYTES"

    # the child stages its parent's physical key from the parent output bucket
    s3.objects[(OUT, physical)] = s3.uploads[f"{OUT}/{physical}"]
    child = _ctx(recorded.output_keys, {STAGE_REQUEST_METADATA_KEY: _req(step="child")})
    child_plan = build_stage_plan(child, s3)
    assert [(o.bucket, o.key) for o in child_plan.objects] == [(OUT, physical)]


def test_a_step_symlink_in_out_never_reaches_the_store(tmp_path):
    out = tmp_path / "out"
    (out / OUT).mkdir(parents=True)
    (out / OUT / "ok.txt").write_text("fine")
    (out / OUT / "ssh").symlink_to("/etc/hostname")
    s3 = _FakeS3()
    with pytest.raises(StepExecutionError, match="a symlink"):
        collect_out(_helper_collect(out), s3, OUT, "wf-1", "seg")
    assert all("ssh" not in key for key in s3.uploads)
