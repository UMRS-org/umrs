"""Embedding batching and I/O-compute pipelining (plan U1 A4).

Port of upstream c321d4f3, 898aac2d, 5894c924, e7b09f59, 384be72f and the
core part of 30ad54f4: single-item calls go through the batch path,
``UMRS_INFERENCE_BATCH_SIZE`` caps each forward pass, the S3 load of batch
N+1 overlaps the compute of batch N, a loader failure reaches the caller and
no prefetch thread outlives the call. Hermetic: S3 loads and the model are
fakes; nothing touches the network or a database.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import Future

import pytest
from umrs.inference import embedding_backend, embeddings

_PREFETCH_THREAD = "umrs-embed-prefetch"


class _FakeModel:
    model_name = "fake-model"
    type = None
    return_type = "embedding"

    def __init__(self, on_call=None):
        self.calls: list[list] = []
        self._on_call = on_call

    def compute_embeddings(self, objects):
        self.calls.append(list(objects))
        if self._on_call is not None:
            self._on_call(len(self.calls))
        return [f"emb:{obj}" for obj in objects]


class _Loader:
    """Stands in for ``load_data_from_s3``: one object per key, in order."""

    def __init__(self, on_load=None):
        self.calls: list[tuple[str, list]] = []
        self._on_load = on_load

    def __call__(self, _s3, bucket, keys, _modality):
        self.calls.append((bucket, list(keys)))
        if self._on_load is not None:
            self._on_load(len(self.calls), bucket, keys)
        objects = [f"{bucket}/{key}" for key in keys]
        return objects, [{} for _ in keys], [f"etag-{k}" for k in keys], [1] * len(keys), False


class _Store:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, _data_base, embeddings, _table_name, bucket, keys, saved,
                 _modality, checksums, _model_name, **_kwargs):
        self.calls.append({
            "bucket": bucket, "keys": list(keys), "embeddings": list(embeddings),
            "checksums": list(checksums), "saved": list(saved),
        })


def _item(bucket, n_keys, model):
    return {
        "path_to_bucket": bucket,
        "s3_keys": [f"k{i}" for i in range(n_keys)],
        "data_base": None,
        "table_name": f"t_{bucket}",
        "model": model,
        "device": "cpu",
        "modality_name": "image",
        "s3_connection": object(),
        "output_bucket": None,
        "task": None,
        "skip_existing": True,
    }


@pytest.fixture
def loader(monkeypatch):
    fake = _Loader()
    monkeypatch.setattr(embeddings, "load_data_from_s3", fake)
    return fake


def _prefetch_threads():
    return [t for t in threading.enumerate() if t.name == _PREFETCH_THREAD]


def test_batch_size_caps_each_forward_pass(monkeypatch, loader):
    monkeypatch.setenv("UMRS_INFERENCE_BATCH_SIZE", "4")
    model, store = _FakeModel(), _Store()
    items = [_item("a", 3, model), _item("b", 4, model), _item("c", 3, model)]

    results = embeddings.compute_and_store_embedding_batch(items, store_fn=store)

    assert results == [True, True, True]
    assert len(model.calls) == math.ceil(10 / 4)
    assert [len(call) for call in model.calls] == [4, 4, 2]
    # Each item is stored once, its embeddings in key order.
    assert [c["bucket"] for c in store.calls] == ["a", "b", "c"]
    for call, item in zip(store.calls, items, strict=True):
        expected = [f"emb:{item['path_to_bucket']}/{k}" for k in item["s3_keys"]]
        assert call["embeddings"] == expected
        assert call["checksums"] == [f"etag-{k}" for k in item["s3_keys"]]
        assert call["saved"] == [None] * len(item["s3_keys"])


def test_unset_knob_keeps_one_forward_pass(monkeypatch, loader):
    monkeypatch.delenv("UMRS_INFERENCE_BATCH_SIZE", raising=False)
    model, store = _FakeModel(), _Store()
    items = [_item("a", 3, model), _item("b", 4, model)]

    assert embeddings.compute_and_store_embedding_batch(items, store_fn=store) == [True, True]
    assert len(model.calls) == 1
    assert len(model.calls[0]) == 7
    assert store.calls[1]["embeddings"] == [f"emb:b/k{i}" for i in range(4)]


@pytest.mark.parametrize("raw", ["abc", "-3", "2.5"])
def test_invalid_knob_falls_back_with_warning(monkeypatch, caplog, raw):
    monkeypatch.setenv("UMRS_INFERENCE_BATCH_SIZE", raw)
    with caplog.at_level(logging.WARNING, logger=embeddings.logger.name):
        assert embeddings.inference_batch_size() == 0
    assert any("UMRS_INFERENCE_BATCH_SIZE" in r.getMessage() for r in caplog.records)


def test_valid_knob_values(monkeypatch):
    monkeypatch.setenv("UMRS_INFERENCE_BATCH_SIZE", " 64 ")
    assert embeddings.inference_batch_size() == 64
    monkeypatch.setenv("UMRS_INFERENCE_BATCH_SIZE", "0")
    assert embeddings.inference_batch_size() == 0


def test_single_item_goes_through_the_batch_path(monkeypatch):
    seen: list[list] = []

    def fake_batch(items):
        seen.append(items)
        return [True]

    monkeypatch.setattr(embedding_backend, "compute_and_store_embedding_batch", fake_batch)
    backend = embedding_backend.LocalEmbeddingBackend()
    ok = backend.compute_and_store(
        "bucket", ["k0"], "db", "tbl", "model", "cpu", "image", "s3", None,
        task="embedding", skip_existing=False,
    )

    assert ok is True
    assert len(seen) == 1 and len(seen[0]) == 1
    item = seen[0][0]
    assert item["path_to_bucket"] == "bucket" and item["s3_keys"] == ["k0"]
    assert item["task"] == "embedding" and item["skip_existing"] is False


def test_single_item_is_batched_by_the_knob(monkeypatch, loader):
    monkeypatch.setenv("UMRS_INFERENCE_BATCH_SIZE", "2")
    model, store = _FakeModel(), _Store()
    monkeypatch.setattr(embeddings, "_resolve_store_fn", lambda _fn: store)
    backend = embedding_backend.LocalEmbeddingBackend()

    ok = backend.compute_and_store(
        "bucket", [f"k{i}" for i in range(5)], None, "tbl", model, "cpu", "image",
        object(), None,
    )

    assert ok is True
    assert [len(c) for c in model.calls] == [2, 2, 1]
    assert store.calls[0]["embeddings"] == [f"emb:bucket/k{i}" for i in range(5)]


def test_single_item_failure_returns_false(monkeypatch):
    monkeypatch.setattr(
        embedding_backend, "compute_and_store_embedding_batch", lambda _items: [False]
    )
    backend = embedding_backend.LocalEmbeddingBackend()
    assert backend.compute_and_store("b", ["k"], None, "t", None, "cpu", "image", None, None) is False


def test_pipeline_loads_next_batch_while_computing(monkeypatch):
    """Batch 2 is loaded while batch 1 is still in the forward pass."""
    monkeypatch.setenv("UMRS_INFERENCE_BATCH_SIZE", "2")
    second_loaded = threading.Event()
    overlapped: list[bool] = []

    def on_load(n, bucket, keys):
        if n == 2:
            second_loaded.set()

    def on_call(n):
        if n == 1:
            overlapped.append(second_loaded.wait(timeout=5))

    monkeypatch.setattr(embeddings, "load_data_from_s3", _Loader(on_load))
    model, store = _FakeModel(on_call), _Store()

    embeddings.compute_and_store_embedding_batch([_item("a", 6, model)], store_fn=store)

    assert overlapped == [True]


def test_pipeline_preserves_order_under_jittery_io(monkeypatch):
    monkeypatch.setenv("UMRS_INFERENCE_BATCH_SIZE", "3")

    def on_load(n, bucket, keys):
        time.sleep(0.002 * (n % 3))

    monkeypatch.setattr(embeddings, "load_data_from_s3", _Loader(on_load))
    model, store = _FakeModel(), _Store()
    items = [_item("a", 5, model), _item("b", 1, model), _item("c", 7, model)]

    embeddings.compute_and_store_embedding_batch(items, store_fn=store)

    flat = [obj for call in model.calls for obj in call]
    assert flat == [f"{i['path_to_bucket']}/{k}" for i in items for k in i["s3_keys"]]
    for call, item in zip(store.calls, items, strict=True):
        assert call["embeddings"] == [f"emb:{item['path_to_bucket']}/{k}" for k in item["s3_keys"]]


def test_loader_failure_propagates_and_leaves_no_thread(monkeypatch):
    monkeypatch.setenv("UMRS_INFERENCE_BATCH_SIZE", "2")

    def on_load(n, bucket, keys):
        if n == 2:
            raise ConnectionError("s3 down")

    monkeypatch.setattr(embeddings, "load_data_from_s3", _Loader(on_load))
    model, store = _FakeModel(), _Store()

    with pytest.raises(ConnectionError, match="s3 down"):
        embeddings.compute_and_store_embedding_batch([_item("a", 8, model)], store_fn=store)
    assert store.calls == []
    assert _prefetch_threads() == []


def test_model_failure_stops_a_blocked_loader(monkeypatch):
    """The loader is parked on a full queue when compute fails: it must exit."""
    monkeypatch.setenv("UMRS_INFERENCE_BATCH_SIZE", "1")
    monkeypatch.setattr(embeddings, "load_data_from_s3", _Loader())

    def on_call(n):
        time.sleep(0.05)  # let the loader fill the bounded queue
        raise RuntimeError("cuda oom")

    model = _FakeModel(on_call)
    with pytest.raises(RuntimeError, match="cuda oom"):
        embeddings.compute_and_store_embedding_batch([_item("a", 50, model)], store_fn=_Store())
    assert _prefetch_threads() == []


def test_timing_is_logged_at_debug(monkeypatch, loader, caplog):
    monkeypatch.setenv("UMRS_INFERENCE_BATCH_SIZE", "2")
    model = _FakeModel()
    with caplog.at_level(logging.DEBUG, logger=embeddings.logger.name):
        embeddings.compute_and_store_embedding_batch([_item("a", 3, model)], store_fn=_Store())
    timing = [r for r in caplog.records if hasattr(r, "umrs_embed_timing")]
    assert len(timing) == 1
    assert timing[0].levelno == logging.DEBUG
    assert timing[0].umrs_embed_timing["n_objects"] == 3
    assert timing[0].umrs_embed_timing["batch_size"] == 2


def test_no_objects_reports_failure(monkeypatch, loader):
    model = _FakeModel()
    assert embeddings.compute_and_store_embedding_batch(
        [_item("a", 0, model)], store_fn=_Store()
    ) == [False]
    assert model.calls == []


def test_pool_task_failure_logs_traceback(tmp_path, caplog):
    from diskcache import Cache
    from umrs.core.UMRSThreads import UMRSThreads

    threads = UMRSThreads(Cache(str(tmp_path / "s")), Cache(str(tmp_path / "t")))
    try:
        failed: Future = Future()
        try:
            raise ValueError("boom")
        except ValueError as exc:
            failed.set_exception(exc)
        threads.thread_list.append(failed)
        with caplog.at_level(logging.ERROR, logger="umrs.core.UMRSThreads"):
            threads.clean_finished_threads()
    finally:
        threads.close()
    records = [r for r in caplog.records if "Pool task raised" in r.getMessage()]
    assert records and records[0].exc_info is not None
