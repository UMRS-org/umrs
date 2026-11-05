"""ADR-109 D-1/D-2 (V3): the svc-vision2 role probe and the sidecar budget.

The role probe of a vision co-serving is one segmentation of a small image
the service generates itself (no file ships), with at least one mask back.
The sidecar keeps to its budget beside vLLM: VISION_MAX_BATCH inferences at
once and a CUDA memory cap.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from umrs.orchestrator.readiness import ReadinessConfig, ReadinessPoller

SRC = Path(__file__).resolve().parents[1] / "src/umrs"


def _poller(handler) -> ReadinessPoller:
    poller = ReadinessPoller(ReadinessConfig(timeout_sec=5))
    poller._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return poller


def _service_handler(probe_body, seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.content))
        if request.url.path == "/readyz":
            return httpx.Response(200)
        return httpx.Response(200, json=probe_body)

    return handler


async def test_segment_probe_passes_with_a_mask():
    seen: list = []
    poller = _poller(_service_handler({"masks": 2}, seen))
    ok, degraded, models, error = await poller._check_once(
        "gb10-01", 12434, probe_model="sam3", probe_kind="segment", probe_path="/readyz"
    )
    assert (ok, degraded, error) == (True, False, "")
    assert [(m, p) for m, p, _ in seen] == [("GET", "/readyz"), ("POST", "/probe")]
    assert b'"model_type":"sam3"' in seen[1][2].replace(b" ", b"")


@pytest.mark.parametrize(
    "body", [{"masks": 0}, {"masks": 0, "error": "CUDA out of memory"}, ["x"]]
)
async def test_segment_probe_without_a_mask_is_degraded(body):
    poller = _poller(_service_handler(body, []))
    ok, degraded, _models, error = await poller._check_once(
        "gb10-01", 12434, probe_model="sam3", probe_kind="segment"
    )
    assert (ok, degraded) == (False, True)
    assert "no mask" in error


async def test_http_probe_still_needs_only_readyz():
    seen: list = []
    poller = _poller(_service_handler({}, seen))
    ok, _degraded, _models, _error = await poller._check_once(
        "gb10-01", 12434, probe_kind="http", probe_path="/readyz"
    )
    assert ok and [p for _m, p, _c in seen] == ["/readyz"]


# ── the vision service side ─────────────────────────────────────────────


def _service_module():
    pytest.importorskip("bentoml")
    pytest.importorskip("cv2")
    from umrs.vision_service import service

    return service


def test_probe_image_is_generated_and_small():
    from umrs.vision_service.default_adapters.sam3_adapter import probe_image

    image = probe_image()
    assert image.size == (256, 256)
    assert image.getpixel((128, 128)) == (220, 20, 20)
    assert image.getpixel((2, 2)) == (128, 128, 128)


def test_sam3_adapter_probe_counts_masks(monkeypatch):
    import umrs.inference.sam3_segmentation as seg
    from umrs.vision_service.default_adapters.sam3_adapter import SAM3Adapter

    calls = []

    def segment(model, processor, image, prompt, device, threshold, mask_threshold):
        calls.append((image.size, prompt))
        return {"num_objects": 1, "masks": [object()], "boxes": [], "scores": []}

    monkeypatch.setattr(seg, "segment_image_with_text", segment)
    result = SAM3Adapter().probe({"model": 1, "processor": 2, "device": "cpu"})
    assert result["masks"] == 1
    assert calls == [((256, 256), "red circle")]


def test_run_probe_reports_errors_without_raising():
    service = _service_module()

    class Adapter:
        def probe(self, _model):
            raise RuntimeError("weights missing")

    class Cache:
        def get_or_load(self, _adapter, _config):
            return object()

    assert service.run_probe({"sam3": Adapter()}, Cache(), "sam3") == {
        "masks": 0,
        "error": "RuntimeError: weights missing",
    }
    assert service.run_probe({}, Cache(), "sam3")["masks"] == 0


def test_gpu_budget_caps_the_process_fraction(monkeypatch):
    service = _service_module()
    calls = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _i: SimpleNamespace(total_memory=128 << 30),
        set_per_process_memory_fraction=lambda f, d: calls.append((f, d)),
    )
    monkeypatch.setenv("VISION_GPU_MEMORY_BUDGET_GIB", "16")
    assert service.apply_gpu_memory_budget(SimpleNamespace(cuda=cuda)) == 0.125
    assert calls == [(0.125, 0)]
    monkeypatch.delenv("VISION_GPU_MEMORY_BUDGET_GIB")
    assert service.apply_gpu_memory_budget(SimpleNamespace(cuda=cuda)) is None
    assert len(calls) == 1


def test_max_batch_serialises_inference(monkeypatch):
    service = _service_module()
    monkeypatch.setenv("VISION_MAX_BATCH", "1")
    running, peak = [0], [0]
    lock = threading.Lock()

    class Adapter:
        def infer(self, *_a):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.02)
            with lock:
                running[0] -= 1

    adapters = {"a": Adapter(), "b": Adapter()}
    service.bound_concurrency(adapters, service.max_batch())
    service.bound_concurrency(adapters, service.max_batch())  # idempotent
    threads = [
        threading.Thread(target=adapters[k].infer) for k in ("a", "b", "a", "b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak[0] == 1


def test_the_compose_and_wrapper_carry_the_budget_to_the_sidecar():
    compose = (SRC / "mini-rivault-llm/serve-model.yaml").read_text(encoding="utf-8")
    wrapper = (SRC / "mini-rivault-llm/serve-model.slurm").read_text(encoding="utf-8")
    assert "PYTORCH_CUDA_ALLOC_CONF: ${VISION_PYTORCH_CUDA_ALLOC_CONF:-}" in compose
    assert "VISION_MAX_BATCH: ${VISION_MAX_BATCH:-}" in compose
    env_file = wrapper.split('cat <<EOF > "./env.${uuid}"', 1)[1].split("\nEOF\n", 1)[0]
    for key in (
        "VISION_MAX_BATCH",
        "VISION_GPU_MEMORY_BUDGET_GIB",
        "VISION_PYTORCH_CUDA_ALLOC_CONF",
        "MAX_VISION_MODELS",
    ):
        assert f"\n{key}=${{{key}:-" in env_file, key
