from botocore.exceptions import ClientError
from umrs.vision_service import bentoml_adapter


class FakeS3:
    def head_object(self, Bucket, Key):
        if Bucket == "output-bucket":
            raise ClientError(
                {"Error": {"Code": "404", "Message": "not found"}},
                "HeadObject",
            )
        return {"ETag": '"abc123"', "Metadata": {"sample": "yes"}}


def _patch_bentoml(monkeypatch, calls):
    async def fake_get_or_open_client(service_url, token):
        calls["client"] = {"service_url": service_url, "token": token}
        return object()

    async def fake_submit_and_wait(client, endpoint_name, **kwargs):
        calls["endpoint_name"] = endpoint_name
        calls["kwargs"] = kwargs
        return {"status": "success"}

    monkeypatch.setattr(bentoml_adapter, "_get_or_open_client", fake_get_or_open_client)
    monkeypatch.setattr(bentoml_adapter, "async_submit_and_wait", fake_submit_and_wait)
    monkeypatch.setattr(
        bentoml_adapter,
        "generate_presigned_get",
        lambda *args, **kwargs: "http://s3.example/input.png",
    )
    monkeypatch.setattr(
        bentoml_adapter,
        "generate_presigned_put",
        lambda *args, **kwargs: "http://s3.example/output.pkl",
    )


def test_jens_adapter_calls_vision_task_with_sam3_prompt(monkeypatch):
    calls = {}
    _patch_bentoml(monkeypatch, calls)

    compute = bentoml_adapter.make_jens_vision_compute(
        "http://vision.example",
        "sam3",
        {"prompt": "crack"},
        token="secret-token",
    )

    results, checksums, metadata, saved_keys = compute(
        None,
        FakeS3(),
        ["images/cat.png"],
        "input-bucket",
        "output-bucket",
    )

    assert calls["client"] == {
        "service_url": "http://vision.example",
        "token": "secret-token",
    }
    assert calls["endpoint_name"] == "vision_task"
    assert calls["kwargs"]["image"] == "http://s3.example/input.png"
    assert calls["kwargs"]["out"] == "http://s3.example/output.pkl"
    assert calls["kwargs"]["model"] == "sam3"
    assert calls["kwargs"]["task"] == "crack"
    assert checksums == ["abc123"]
    assert metadata == [{"sample": "yes"}]
    assert saved_keys == ["s3://output-bucket/images/cat_sam3_results.pkl"]
    assert results[0]["results_pkl_s3_key"] == saved_keys[0]


def test_jens_adapter_maps_hmae_mae_task(monkeypatch):
    calls = {}
    _patch_bentoml(monkeypatch, calls)

    compute = bentoml_adapter.make_jens_vision_compute(
        "http://vision.example",
        "hmae",
        {"task": "mae_reconstruction"},
    )

    results, _, _, saved_keys = compute(
        None,
        FakeS3(),
        ["ct/volume_170.png"],
        "input-bucket",
        "output-bucket",
    )

    assert calls["endpoint_name"] == "vision_task"
    assert calls["kwargs"]["model"] == "hmae"
    assert calls["kwargs"]["task"] == "mae"
    assert saved_keys == ["s3://output-bucket/ct/volume_170_mae_results.pkl"]
    assert results[0]["task"] == "mae"
