"""Step worker JSON mode (ADR-112 wave B2; plan §5 B2.0 items 8 and 9).

On the gated lane the gate writes ``input.json`` from signed fields and the
worker reads it as data: the entrypoint must equal the signed
``UMRS_STEP_ENTRYPOINT`` before anything is imported, parent results are
rebuilt with ``step_reply.checked_result``, ``pickle.loads`` is never called
and the reply is the JSON reply the staged lane writes today. Pool, ZMQ and
one-shot pickle modes are unchanged.
"""

from __future__ import annotations

import functools
import json
import os
import pickle
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest
from umrs.pipeline_sdk.staged_s3 import StagedS3Client
from umrs.pipeline_sdk.step import StepResult
from umrs.server import step_worker
from umrs.server.step_reply import result_from_json

UMRS_ROOT = Path(__file__).resolve().parents[1]
CONTAINERFILE = UMRS_ROOT / "docker" / "Containerfile.step"
RUN_ID = "wf-0123456789ab"
OUT = "wf-runner-output"
PARENT_KEY = f"runs/{RUN_ID}/prep/frame.bin"
CONFIG = {"threshold": 0.25, "label": None, "n": 3, "nested": {"t": 1.5}}
_ENVS = (
    "UMRS_STEP_INPUT",
    "UMRS_STEP_INPUT_S3",
    "UMRS_STEP_OUTPUT",
    "UMRS_STEP_OUTPUT_S3",
    "UMRS_STEP_OUTPUT_JSON",
    "UMRS_STEP_INPUT_JSON",
    "UMRS_STEP_ENTRYPOINT",
    "UMRS_STEP_STAGED",
    "UMRS_STEP_SOURCE_DIR",
)

STEP_SOURCE = """
from umrs.pipeline_sdk.step import Step, StepContext, StepResult


class Echo(Step):
    name = "echo"

    def setup(self, config):
        self.cfg = config

    def run(self, ctx: StepContext) -> StepResult:
        data = ctx.download_parent_bytes(ctx.input_keys[0])
        ctx.upload_bytes("echo.bin", data[::-1])
        parent = ctx.parent_results["prep"]
        return StepResult(
            output_keys=["echo.bin"],
            extra={
                "config": self.cfg,
                "ctx_config": ctx.config,
                "run_id": ctx.run_id,
                "metadata": ctx.metadata,
                "s3": type(ctx.s3).__name__,
                "input_keys": ctx.input_keys,
                "parent_type": type(parent).__name__,
                "parent_keys": parent.output_keys,
                "parent_extra": parent.extra,
                "parent_bucket": ctx.parent_output_bucket,
            },
        )
"""

SENTINEL_SOURCE = """
import os
open(os.environ["SENTINEL_PATH"], "w").close()
"""

SUBPROCESS_WORKER = """import functools, pickle, sys
wd = sys.argv[1]


def _never(*_a, **_k):
    raise AssertionError("pickle.loads called in JSON mode")


pickle.loads = _never
from umrs.pipeline_sdk import staged_s3
staged_s3.StagedS3Client.__init__.__defaults__ = (f"{wd}/in", f"{wd}/out", None)
from umrs.server import step_worker
sys.exit(step_worker.entry(sys.argv[2:]))
"""


def _encode(body: object) -> bytes:
    """The gate's encoding of ``input.json`` (plan §5 B2.0 item 8)."""
    text = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


def _body(entrypoint: str) -> dict:
    return {
        "v": 1,
        "entrypoint": entrypoint,
        "config": dict(CONFIG),
        "parent_results": {
            "prep": {
                "output_keys": [f"s3://{OUT}/{PARENT_KEY}"],
                "checksums": ["c1"],
                "metadata": [{"k": None}],
                "extra": {"score": 0.5},
                "embeddings": None,
            }
        },
        "ctx": {
            "run_tag": RUN_ID,
            "step": "echo",
            "attempt": 1,
            "input_bucket": "wf-runner",
            "output_bucket": OUT,
            "parent_output_bucket": OUT,
            "input_keys": [PARENT_KEY],
        },
    }


def _workdir(tmp_path: Path, package: str) -> Path:
    """A gate-style workdir: src/ with one step package, in/ with the
    manifest and one parent blob, empty out/ and reply/."""
    wd = tmp_path / "wd"
    pkg = wd / "src" / package
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "echo.py").write_text(STEP_SOURCE)
    (pkg / "boom.py").write_text(SENTINEL_SOURCE)
    blob = wd / "in" / "blobs" / ("a" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"abc")
    manifest = {
        "objects": {f"{OUT}/{PARENT_KEY}": {"path": f"blobs/{'a' * 64}", "size": 3}},
        "prefixes": [],
        "output_bucket": OUT,
    }
    (wd / "in" / "manifest.json").write_text(json.dumps(manifest))
    (wd / "out").mkdir()
    (wd / "reply").mkdir()
    return wd


@pytest.fixture
def gated(tmp_path, monkeypatch):
    """A gated workdir, the job-script env, a fresh step package name and
    ``pickle.loads`` patched to fail the test if anything calls it."""
    package = f"wfsteps_{uuid.uuid4().hex[:12]}"
    wd = _workdir(tmp_path, package)
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)
    entrypoint = f"{package}.echo:Echo"
    monkeypatch.setenv("UMRS_STEP_INPUT_JSON", str(wd / "input.json"))
    monkeypatch.setenv("UMRS_STEP_ENTRYPOINT", entrypoint)
    monkeypatch.setenv("UMRS_STEP_OUTPUT_JSON", str(wd / "reply" / "output.json"))
    monkeypatch.setenv("UMRS_STEP_STAGED", "1")
    monkeypatch.setenv("UMRS_STEP_SOURCE_DIR", str(wd / "src"))
    monkeypatch.setenv("SENTINEL_PATH", str(tmp_path / "imported"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(step_worker, "_WORKER_SOURCE_DIRS", [])
    monkeypatch.setattr(
        step_worker,
        "StagedS3Client",
        functools.partial(StagedS3Client, str(wd / "in"), str(wd / "out")),
    )

    def _never(*_a, **_k):
        raise AssertionError("pickle.loads called in JSON mode")

    monkeypatch.setattr(pickle, "loads", _never)
    (wd / "input.json").write_bytes(_encode(_body(entrypoint)))
    yield wd, entrypoint, tmp_path / "imported"
    for name in [m for m in sys.modules if m.split(".")[0] == package]:
        del sys.modules[name]


def _reply(wd: Path) -> dict:
    return json.loads((wd / "reply" / "output.json").read_bytes())


def _refused(wd: Path, name: str) -> str:
    reply = _reply(wd)
    assert reply["status"] == "error" and reply["phase"] == "worker", reply
    assert reply["message"].startswith(f"{name}: "), reply["message"]
    return reply["message"]


# ── round trip ──────────────────────────────────────────────────────────────


def _run_worker_process(wd: Path) -> subprocess.CompletedProcess:
    """The worker in its own process with only the job-script variables."""
    env = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
    env.update({k: os.environ[k] for k in _ENVS if k in os.environ})
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", SUBPROCESS_WORKER, str(wd)],
        env=env,
        capture_output=True,
        timeout=120,
    )


EXPECTED_EXTRA = {
    "config": CONFIG,
    "ctx_config": CONFIG,
    "run_id": RUN_ID,
    "metadata": {},
    "s3": "StagedS3Client",
    "input_keys": [PARENT_KEY],
    "parent_type": "StepResult",
    "parent_keys": [f"s3://{OUT}/{PARENT_KEY}"],
    "parent_extra": {"score": 0.5},
    "parent_bucket": OUT,
}


def test_round_trip_in_a_real_process(gated):
    """A worker process: a real Step class from the src dir, the staged
    in/out, the JSON reply the server reads, no pickle at all."""
    wd, _, _ = gated
    proc = _run_worker_process(wd)
    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout == b""  # no pickle reply on stdout
    result = result_from_json((wd / "reply" / "output.json").read_bytes())
    assert result.output_keys == ["echo.bin"]
    assert result.extra == EXPECTED_EXTRA
    assert (wd / "out" / OUT / "echo.bin").read_bytes() == b"cba"
    assert sorted(p.name for p in (wd / "reply").iterdir()) == ["output.json"]


def test_floats_and_null_reach_the_step_unchanged(gated):
    wd, _, _ = gated
    assert step_worker.main() == 0
    reply = _reply(wd)
    assert reply["status"] == "ok", reply
    config = reply["result"]["extra"]["config"]
    assert config == CONFIG
    assert type(config["threshold"]) is float and config["label"] is None


def test_json_mode_writes_no_pickle_even_unstaged(gated, monkeypatch, capsysbinary):
    """The reply is JSON only: without ``UMRS_STEP_STAGED`` the one-shot
    pickle fallback to stdout stays off in JSON mode."""
    wd, _, _ = gated
    monkeypatch.delenv("UMRS_STEP_STAGED")
    assert step_worker.main() == 0
    assert _reply(wd)["status"] == "ok"
    assert capsysbinary.readouterr().out == b""


# ── refusals (each before the step module is imported) ──────────────────────


def test_entrypoint_mismatch_is_refused_before_import(gated, monkeypatch):
    wd, entrypoint, sentinel = gated
    package = entrypoint.split(".")[0]
    body = _body(f"{package}.boom:Boom")
    (wd / "input.json").write_bytes(_encode(body))
    assert step_worker.main() == 0
    assert "boom" in _refused(wd, "entrypoint_mismatch")
    assert not sentinel.exists()
    assert not any(m.startswith(package) for m in sys.modules)


def test_unset_signed_entrypoint_is_a_mismatch(gated, monkeypatch):
    wd, _, sentinel = gated
    monkeypatch.delenv("UMRS_STEP_ENTRYPOINT")
    assert step_worker.main() == 0
    _refused(wd, "entrypoint_mismatch")


def test_pickle_bytes_in_input_json_are_refused(gated):
    wd, entrypoint, sentinel = gated
    (wd / "input.json").write_bytes(pickle.dumps(_body(entrypoint)))
    assert step_worker.main() == 0
    _refused(wd, "input_not_json")
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "text",
    [
        '{"v":1,"v":1}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"x":1e999}',
        "",
    ],
)
def test_strict_parse_refuses_non_json(gated, text):
    wd, _, _ = gated
    (wd / "input.json").write_text(text)
    assert step_worker.main() == 0
    _refused(wd, "input_not_json")


def _mutate(body: dict, path: tuple, value: object) -> dict:
    target = body
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return body


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("extra",), 1),
        (("v",), 2),
        (("v",), True),
        (("entrypoint",), ""),
        (("entrypoint",), 7),
        (("config",), []),
        (("parent_results",), None),
        (("ctx",), []),
        (("ctx", "attempt"), True),
        (("ctx", "attempt"), "1"),
        (("ctx", "run_tag"), None),
        (("ctx", "input_keys"), [1]),
        (("ctx", "input_keys"), "k"),
        (("ctx", "step"), KeyError),
        (("ctx", "extra"), "x"),
        (("ctx",), KeyError),
    ],
)
def test_schema_refusals(gated, path, value):
    wd, entrypoint, sentinel = gated
    body = _mutate(_body(entrypoint), path, value)
    (wd / "input.json").write_bytes(_encode(body))
    assert step_worker.main() == 0
    _refused(wd, "input_schema")
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "parent",
    [
        {"output_keys": [1]},
        {"output_keys": "s3://x/y"},
        {"extra": []},
        {"embeddings": "nope"},
        ["not", "an", "object"],
    ],
)
def test_parent_results_are_rebuilt_with_checked_result(gated, parent):
    wd, entrypoint, _ = gated
    body = _body(entrypoint)
    body["parent_results"]["prep"] = parent
    (wd / "input.json").write_bytes(_encode(body))
    assert step_worker.main() == 0
    assert "prep" in _refused(wd, "parent_result_invalid")


def test_missing_input_json_is_unreadable(gated):
    wd, _, _ = gated
    (wd / "input.json").unlink()
    assert step_worker.main() == 0
    _refused(wd, "input_unreadable")


def test_input_json_over_the_read_cap_is_unreadable(gated, monkeypatch):
    wd, _, _ = gated
    size = (wd / "input.json").stat().st_size
    monkeypatch.setattr(step_worker, "INPUT_JSON_READ_MAX_BYTES", size - 1)
    assert step_worker.main() == 0
    _refused(wd, "input_unreadable")


def test_read_cap_is_64_mib():
    assert step_worker.INPUT_JSON_READ_MAX_BYTES == 64 * 1024 * 1024


def test_missing_staged_manifest_is_refused_by_name(gated):
    wd, _, sentinel = gated
    (wd / "in" / "manifest.json").unlink()
    assert step_worker.main() == 0
    assert "manifest.json" in _refused(wd, "input_unreadable")
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "name",
    ["UMRS_STEP_INPUT", "UMRS_STEP_INPUT_S3", "UMRS_STEP_OUTPUT", "UMRS_STEP_OUTPUT_S3"],
)
def test_pickle_channels_with_json_mode_are_mixed_modes(gated, monkeypatch, name):
    wd, _, _ = gated
    out = wd / "output.pkl"
    monkeypatch.setenv(name, str(out) if name == "UMRS_STEP_OUTPUT" else "b/k")
    assert step_worker.main() == 0
    assert name in _refused(wd, "mixed_input_modes")
    assert not out.exists()  # the reply is JSON only


@pytest.mark.parametrize(
    "argv", [["--pool"], ["--zmq-pool", "tcp://127.0.0.1:1"], ["--zmq-dealer", "x"]]
)
def test_pool_flags_with_json_mode_are_mixed_modes(gated, monkeypatch, argv):
    wd, _, _ = gated

    def _loop(*_a):
        raise AssertionError("a pickle loop started in JSON mode")

    for loop in ("pool_main", "zmq_pool_main", "zmq_dealer_main"):
        monkeypatch.setattr(step_worker, loop, _loop)
    assert step_worker.entry(argv) == 0
    assert argv[0] in _refused(wd, "mixed_input_modes")


def test_missing_output_json_is_mixed_modes_and_exit_1(gated, monkeypatch, capsysbinary):
    wd, _, sentinel = gated
    monkeypatch.delenv("UMRS_STEP_OUTPUT_JSON")
    assert step_worker.main() == 1
    assert not (wd / "reply" / "output.json").exists()
    assert capsysbinary.readouterr().out == b""
    assert not sentinel.exists()


def test_empty_input_json_variable_never_falls_back_to_pickle(gated, monkeypatch):
    wd, _, _ = gated
    monkeypatch.setenv("UMRS_STEP_INPUT_JSON", "")
    assert step_worker.main() == 0
    _refused(wd, "input_unreadable")


# ── pickle modes unchanged ──────────────────────────────────────────────────


def _ok_step_module(tmp_path: Path) -> str:
    package = f"pkl_{uuid.uuid4().hex[:12]}"
    (tmp_path / package).mkdir()
    (tmp_path / package / "__init__.py").write_text(
        textwrap.dedent(
            """
            from umrs.pipeline_sdk.step import Step, StepResult

            class Ok(Step):
                name = "ok"

                def run(self, ctx):
                    return StepResult(output_keys=[ctx.run_id])
            """
        )
    )
    return package


def test_one_shot_pickle_mode_is_unchanged(tmp_path, monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)
    package = _ok_step_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    payload = {
        "entrypoint": f"{package}:Ok",
        "config": {},
        "ctx": {
            "input_bucket": "b",
            "input_keys": [],
            "output_bucket": "b-output",
            "config": {},
            "run_id": "r-1",
        },
    }
    (tmp_path / "input.pkl").write_bytes(pickle.dumps(payload))
    monkeypatch.setenv("UMRS_STEP_INPUT", str(tmp_path / "input.pkl"))
    monkeypatch.setenv("UMRS_STEP_OUTPUT", str(tmp_path / "output.pkl"))
    assert step_worker.entry([]) == 0
    reply = pickle.loads((tmp_path / "output.pkl").read_bytes())
    assert reply["status"] == "ok" and isinstance(reply["result"], StepResult)
    assert reply["result"].output_keys == ["r-1"]
    del sys.modules[package]


@pytest.mark.parametrize("flag", ["--zmq-pool", "--zmq-dealer"])
def test_zmq_flag_without_address_exits_1(monkeypatch, flag):
    monkeypatch.delenv("UMRS_STEP_INPUT_JSON", raising=False)
    assert step_worker.entry([flag]) == 1


@pytest.mark.parametrize(
    ("argv", "loop", "arg"),
    [
        (["--pool"], "pool_main", None),
        (["--zmq-pool", "tcp://127.0.0.1:5"], "zmq_pool_main", "tcp://127.0.0.1:5"),
        (["--zmq-dealer", "tcp://h:6"], "zmq_dealer_main", "tcp://h:6"),
    ],
)
def test_entry_dispatches_pickle_loops_without_json_mode(monkeypatch, argv, loop, arg):
    monkeypatch.delenv("UMRS_STEP_INPUT_JSON", raising=False)
    seen: list = []
    monkeypatch.setattr(step_worker, loop, lambda *a: seen.append(a) or 7)
    assert step_worker.entry(argv) == 7
    assert seen == [(arg,) if arg else ()]


# ── the platform step image ─────────────────────────────────────────────────


def test_step_image_declares_the_json_payload_label():
    text = CONTAINERFILE.read_text()
    assert 'org.umrs.step-payload="pickle,json"' in text
    assert 'org.umrs.step-contract="1"' in text
