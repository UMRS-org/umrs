"""Worker-side staged S3 shim and the stage_inputs manifest field (ADR-074 D-5)."""

from __future__ import annotations

import functools
import io
import json
import pickle

import numpy as np
import pytest
from umrs.pipeline_sdk.staged_s3 import (
    StagedObjectNotStaged,
    StagedS3Client,
    StagedS3Error,
    StagedS3Unsupported,
    normalize_key,
    parse_stage_inputs,
    split_s3_ref,
)
from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.pipeline_sdk.workflow import StepSpec, WorkflowDefinition

OUT = "alice.p.a-output"


def _staged(tmp_path, objects=None, prefixes=()):
    """A /work/in with a server-style manifest, and an empty /work/out."""
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    manifest = {"objects": {}, "prefixes": list(prefixes), "output_bucket": OUT}
    for full, data in (objects or {}).items():
        rel = full
        (in_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (in_dir / rel).write_bytes(data)
        manifest["objects"][full] = {
            "path": rel,
            "size": len(data),
            "etag": '"e"',
            "content_type": "text/plain",
        }
    (in_dir / "manifest.json").write_text(json.dumps(manifest))
    return StagedS3Client(str(in_dir), str(out_dir)), in_dir, out_dir


# ── key / manifest-field rules ──────────────────────────────────────────────


def test_normalize_key_drops_empty_segments_and_refuses_dotdot():
    assert normalize_key("/a//./b/") == "a/b"
    for bad in ("", "a/../b", "..", "a\x00b", "a\nb", "/", "x" * 256):
        with pytest.raises(ValueError):
            normalize_key(bad)


def test_split_s3_ref_uri_and_bare_key():
    assert split_s3_ref("s3://alice.p.a/k/1", "dflt") == ("alice.p.a", "k/1")
    assert split_s3_ref("k/1", "dflt") == ("dflt", "k/1")
    with pytest.raises(ValueError):
        split_s3_ref("s3://Bad_Bucket/k", None)
    with pytest.raises(ValueError):
        split_s3_ref("s3://alice.p.a/", None)


def test_parse_stage_inputs_accepts_and_dedupes():
    parsed = parse_stage_inputs(
        {"params": ["video", "video"], "prefixes": ["s3://alice.p.a/in/", "rel/"]}
    )
    assert parsed == {"params": ["video"], "prefixes": ["s3://alice.p.a/in/", "rel/"]}
    assert parse_stage_inputs(None) == {}


@pytest.mark.parametrize(
    "raw",
    [
        "s3://x/y",
        {"paths": []},
        {"params": "video"},
        {"params": [""]},
        {"prefixes": ["s3://NOPE/x"]},
        {"prefixes": ["a\x01"]},
        {"params": [f"p{i}" for i in range(65)]},
    ],
)
def test_parse_stage_inputs_refuses_malformed_blocks(raw):
    with pytest.raises(ValueError):
        parse_stage_inputs(raw)


def _workflow(step_extra):
    return {
        "name": "wf",
        "source": {"type": "uploaded", "path": "/tmp/wf"},
        "steps": [{"name": "s", "entrypoint": "m:S", **step_extra}],
    }


def test_workflow_manifest_carries_stage_inputs():
    wf = WorkflowDefinition.from_dict(_workflow({"stage_inputs": {"params": ["video"]}}))
    wf.validate()
    assert wf.steps[0].stage_inputs == {"params": ["video"]}
    assert wf.steps[0].to_dict()["stage_inputs"] == {"params": ["video"]}
    # absent stays absent on round-trip
    assert "stage_inputs" not in StepSpec.from_dict({"name": "s", "entrypoint": "m:S"}).to_dict()


def test_upload_validation_refuses_a_malformed_stage_inputs_block():
    # file_handling validates uploads with WorkflowDefinition.from_dict only
    with pytest.raises(ValueError, match="Step 's': stage_inputs"):
        WorkflowDefinition.from_dict(_workflow({"stage_inputs": {"params": "video"}}))


# ── reads ───────────────────────────────────────────────────────────────────


def test_reads_come_from_the_manifest(tmp_path):
    s3, _, _ = _staged(tmp_path, {"alice.p.a/in/v.txt": b"hello"})
    got = s3.get_object(Bucket="alice.p.a", Key="in/v.txt")
    assert got["Body"].read() == b"hello"
    assert got["ContentLength"] == 5
    head = s3.head_object(Bucket="alice.p.a", Key="in/v.txt")
    assert (head["ETag"], head["ContentType"]) == ('"e"', "text/plain")
    dst = tmp_path / "copy"
    s3.download_file("alice.p.a", "in/v.txt", str(dst))
    assert dst.read_bytes() == b"hello"
    buf = io.BytesIO()
    s3.download_fileobj("alice.p.a", "in/v.txt", buf)
    assert buf.getvalue() == b"hello"
    body = s3.get_object(Bucket="alice.p.a", Key="in/v.txt")["Body"]
    assert b"".join(body.iter_chunks(2)) == b"hello"


def test_an_unstaged_object_is_refused_by_name(tmp_path):
    s3, _, _ = _staged(tmp_path, {"alice.p.a/in/v.txt": b"x"})
    with pytest.raises(StagedObjectNotStaged, match="was not staged") as err:
        s3.get_object(Bucket="alice.p.a", Key="in/other.txt")
    assert isinstance(err.value, KeyError)
    assert err.value.response["Error"]["Code"] == "NoSuchKey"
    with pytest.raises(StagedObjectNotStaged):
        s3.head_object(Bucket="bob.p.a", Key="in/v.txt")
    with pytest.raises(StagedObjectNotStaged):
        s3.download_file("alice.p.a", "in/other.txt", str(tmp_path / "dst"))
    assert not (tmp_path / "dst").exists()


def test_range_reads_are_refused_not_ignored(tmp_path):
    s3, _, _ = _staged(tmp_path, {"alice.p.a/k": b"abc"})
    with pytest.raises(StagedS3Error, match="Range"):
        s3.get_object(Bucket="alice.p.a", Key="k", Range="bytes=0-1")


def test_a_malformed_manifest_path_is_refused(tmp_path):
    s3, in_dir, _ = _staged(tmp_path, {"alice.p.a/k": b"abc"})
    manifest = json.loads((in_dir / "manifest.json").read_text())
    manifest["objects"]["alice.p.a/k"]["path"] = "../../etc/passwd"
    s3 = StagedS3Client(str(in_dir), str(tmp_path / "out"), manifest)
    with pytest.raises(StagedS3Error, match="malformed"):
        s3.get_object(Bucket="alice.p.a", Key="k")


def test_missing_manifest_is_a_named_error(tmp_path):
    with pytest.raises(StagedS3Error, match="manifest.json"):
        StagedS3Client(str(tmp_path / "nope"), str(tmp_path / "out"))


# ── writes ──────────────────────────────────────────────────────────────────


def test_writes_land_under_out_bucket_key_normalized(tmp_path):
    s3, _, out_dir = _staged(tmp_path)
    reply = s3.put_object(Bucket=OUT, Key="/res//a.json", Body=b"{}", ContentType="x")
    assert (out_dir / OUT / "res" / "a.json").read_bytes() == b"{}"
    assert reply["ETag"] == '"99914b932bd37a50b983c5e7c90ae93b"'
    s3.put_object(Bucket=OUT, Key="s.txt", Body="text")
    assert (out_dir / OUT / "s.txt").read_text() == "text"
    src = tmp_path / "src.bin"
    src.write_bytes(b"\x00\x01")
    s3.upload_file(str(src), OUT, "up/f.bin")
    s3.upload_fileobj(io.BytesIO(b"stream"), OUT, "up/g.bin")
    assert (out_dir / OUT / "up" / "f.bin").read_bytes() == b"\x00\x01"
    assert (out_dir / OUT / "up" / "g.bin").read_bytes() == b"stream"
    # the step reads its own writes back
    assert s3.get_object(Bucket=OUT, Key="res/a.json")["Body"].read() == b"{}"


def test_writes_elsewhere_are_refused(tmp_path):
    s3, _, out_dir = _staged(tmp_path)
    with pytest.raises(StagedS3Error, match="only to its output bucket"):
        s3.put_object(Bucket="alice.p.a", Key="k", Body=b"x")
    with pytest.raises(StagedS3Error, match=r"'\.\.' segment"):
        s3.put_object(Bucket=OUT, Key="../escape", Body=b"x")
    with pytest.raises(StagedS3Error, match="Body must be"):
        s3.put_object(Bucket=OUT, Key="k", Body=12)
    assert list(out_dir.rglob("*")) == []


def test_step_context_helpers_work_over_the_shim(tmp_path):
    s3, _, out_dir = _staged(tmp_path, {"alice.p.a-output/parent/x.bin": b"px"})
    ctx = StepContext(
        s3=s3,
        input_bucket="alice.p.a",
        input_keys=["parent/x.bin"],
        output_bucket=OUT,
        parent_output_bucket="alice.p.a-output",
    )
    assert ctx.download_parent_bytes("parent/x.bin") == b"px"
    assert ctx.upload_bytes("r/y.bin", b"yy") == f"s3://{OUT}/r/y.bin"
    ctx.upload_numpy("r/a.npy", np.arange(3))
    assert (out_dir / OUT / "r" / "y.bin").read_bytes() == b"yy"
    assert ctx.download_numpy("r/a.npy", bucket=OUT).tolist() == [0, 1, 2]


# ── listing and everything else ─────────────────────────────────────────────


def _listing_client(tmp_path):
    objects = {f"alice.p.a/in/{name}": b"x" for name in ("a", "b", "d/1", "d/2", "e")}
    objects["alice.p.a/other/z"] = b"z"
    return _staged(tmp_path, objects, prefixes=["alice.p.a/in/"])[0]


def test_listing_answers_only_declared_prefixes(tmp_path):
    s3 = _listing_client(tmp_path)
    page = s3.list_objects_v2(Bucket="alice.p.a", Prefix="in/")
    assert [c["Key"] for c in page["Contents"]] == ["in/a", "in/b", "in/d/1", "in/d/2", "in/e"]
    narrower = s3.list_objects_v2(Bucket="alice.p.a", Prefix="in/d/")
    assert narrower["KeyCount"] == 2
    for bucket, prefix in (("alice.p.a", ""), ("alice.p.a", "other/"), ("bob.p.a", "in/")):
        with pytest.raises(StagedS3Error, match="stage_inputs.prefixes"):
            s3.list_objects_v2(Bucket=bucket, Prefix=prefix)


def test_listing_pages_and_folds_delimiters(tmp_path):
    s3 = _listing_client(tmp_path)
    pages = list(
        s3.get_paginator("list_objects_v2").paginate(Bucket="alice.p.a", Prefix="in/", MaxKeys=2)
    )
    keys = [c["Key"] for p in pages for c in p["Contents"]]
    assert keys == ["in/a", "in/b", "in/d/1", "in/d/2", "in/e"]
    assert [p["IsTruncated"] for p in pages] == [True, True, False]
    folded = list(
        s3.get_paginator("list_objects_v2").paginate(
            Bucket="alice.p.a", Prefix="in/", Delimiter="/", MaxKeys=2
        )
    )
    flat = [
        (c.get("Key"), c.get("Prefix"))
        for p in folded
        for c in p["Contents"] + p.get("CommonPrefixes", [])
    ]
    assert sorted(flat, key=str) == sorted(
        [("in/a", None), ("in/b", None), (None, "in/d/"), ("in/e", None)], key=str
    )


def test_other_operations_raise_a_named_error(tmp_path):
    s3, _, _ = _staged(tmp_path)
    assert not hasattr(s3, "delete_object")
    for op in ("delete_object", "copy_object", "list_buckets", "generate_presigned_url"):
        with pytest.raises(StagedS3Unsupported, match="ADR-074 D-5"):
            getattr(s3, op)
    with pytest.raises(StagedS3Unsupported):
        s3.get_paginator("list_buckets")


# ── worker wiring ───────────────────────────────────────────────────────────


class _EchoStep(Step):
    name = "echo"

    def run(self, ctx: StepContext) -> StepResult:
        data = ctx.download_parent_bytes(ctx.input_keys[0])
        ctx.upload_bytes("echo.bin", data[::-1])
        return StepResult(output_keys=["echo.bin"])


def test_worker_uses_the_staged_client_when_staged(tmp_path, monkeypatch):
    from umrs.server import step_worker

    s3, _, out_dir = _staged(tmp_path, {"alice.p.a-output/p.bin": b"abc"})
    monkeypatch.setenv("UMRS_STEP_STAGED", "1")
    monkeypatch.setattr(
        step_worker,
        "StagedS3Client",
        functools.partial(StagedS3Client, str(tmp_path / "in"), str(out_dir)),
    )
    reply = step_worker._process_task(
        {
            "entrypoint": f"{__name__}:_EchoStep",
            "config": {},
            "s3_config": {"endpoint_url": "http://never"},
            "ctx": {
                "input_bucket": "alice.p.a",
                "input_keys": ["p.bin"],
                "output_bucket": OUT,
                "parent_output_bucket": "alice.p.a-output",
                "config": {},
            },
        }
    )
    assert reply["status"] == "ok", reply
    assert (out_dir / OUT / "echo.bin").read_bytes() == b"cba"


def test_staged_worker_without_a_manifest_fails_by_name(monkeypatch):
    from umrs.server import step_worker

    monkeypatch.setenv("UMRS_STEP_STAGED", "1")
    with pytest.raises(StagedS3Error, match="manifest.json"):
        step_worker._build_s3_client({"endpoint_url": "http://127.0.0.1:9000"})


def test_staged_worker_writes_only_the_json_reply(tmp_path, monkeypatch, capsysbinary):
    from umrs.server import step_worker
    from umrs.server.step_reply import result_from_json

    js = tmp_path / "reply" / "output.json"
    js.parent.mkdir()
    monkeypatch.setenv("UMRS_STEP_STAGED", "1")
    monkeypatch.setenv("UMRS_STEP_OUTPUT_JSON", str(js))
    monkeypatch.delenv("UMRS_STEP_OUTPUT", raising=False)
    step_worker._emit({"status": "ok", "result": StepResult(output_keys=["k"])})
    assert result_from_json(js.read_bytes()).output_keys == ["k"]
    assert capsysbinary.readouterr().out == b""  # no pickle on stdout
    assert [p.name for p in js.parent.iterdir()] == ["output.json"]  # no temp left


def test_json_reply_keeps_the_umask_mode(tmp_path, monkeypatch):
    import os

    from umrs.server import step_worker

    js = tmp_path / "output.json"
    monkeypatch.setenv("UMRS_STEP_STAGED", "1")
    monkeypatch.setenv("UMRS_STEP_OUTPUT_JSON", str(js))
    old = os.umask(0o022)
    try:
        step_worker._emit({"status": "ok", "result": StepResult()})
    finally:
        os.umask(old)
    assert js.stat().st_mode & 0o777 == 0o644


def test_unstaged_worker_still_writes_the_pickle(tmp_path, monkeypatch):
    from umrs.server import step_worker

    out = tmp_path / "output.pkl"
    monkeypatch.delenv("UMRS_STEP_STAGED", raising=False)
    monkeypatch.setenv("UMRS_STEP_OUTPUT", str(out))
    monkeypatch.setenv("UMRS_STEP_OUTPUT_JSON", str(tmp_path / "output.json"))
    step_worker._emit({"status": "ok", "result": StepResult()})
    assert pickle.loads(out.read_bytes())["status"] == "ok"
