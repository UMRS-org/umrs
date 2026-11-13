"""``umrs knn`` client action (port of upstream 487e3efb / e04b3d4e / bd0e6bc1; plan U1 A6).

The action uploads the query image to the bucket (under the alias's subdir)
and sends one KNN request; S3 and the control queue are fakes.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

client_umrs = pytest.importorskip("client_umrs")

BUCKET = "alice.xp.images"


class _S3:
    def __init__(self):
        self.uploads = []

    def upload_file(self, path, bucket, key):
        self.uploads.append((path, bucket, key))


class _MQ:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        self.closed = False

    def request_knn(self, client_id, n, s3_object, bucket, subdir=None):
        self.requests.append((client_id, n, s3_object, bucket, subdir))
        return json.dumps(self.reply)

    def close(self):
        self.closed = True


class _Cfg:
    def __init__(self, entries):
        self.entries = entries

    def get(self, alias, **_kwargs):
        return self.entries.get(alias)


_OK = {"status": "ok", "n": 2, "query": "pets/q.png", "bucket": BUCKET,
       "results": [{"object": f"{BUCKET}/pets/a.png", "score": 0.9}],
       "n_indexed": 4, "summary": "KNN results  |  k/n: 2/4"}


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "q.png"
    path.write_bytes(b"png")
    return path


@pytest.fixture
def s3(monkeypatch):
    fake = _S3()
    monkeypatch.setattr(client_umrs, "connect_to_s3", lambda *_a, **_k: (fake, None))
    return fake


def _ctx(image, reply, alias="images", output="text", username=None):
    args = argparse.Namespace(n=2, image_path=str(image), bucket_alias=alias,
                              username=username, output=output, verbose=False)
    entry = SimpleNamespace(path_to_bucket=BUCKET, s3_configuration_file="/nonexistent/s3cfg")
    return SimpleNamespace(args=args, control_mq=_MQ(reply), connection=None,
                           user={"username": "alice"}, buckets_cfg=_Cfg({"images": entry}))


def test_subdir_alias_uploads_under_it_and_scopes_the_search(image, s3, capsys):
    ctx = _ctx(image, _OK, alias="images/pets")

    assert client_umrs._action_knn(ctx) == 0

    assert s3.uploads == [(str(image), BUCKET, "pets/q.png")]
    assert ctx.control_mq.requests == [("alice", 2, "pets/q.png", BUCKET, "pets")]
    assert capsys.readouterr().out.strip() == _OK["summary"]
    assert ctx.control_mq.closed


def test_without_subdir_the_key_is_the_file_name(image, s3):
    ctx = _ctx(image, _OK, username="alice")
    assert client_umrs._action_knn(ctx) == 0
    assert ctx.control_mq.requests == [("alice", 2, "q.png", BUCKET, None)]


def test_json_output_carries_the_reply(image, s3, capsys):
    ctx = _ctx(image, _OK, output="json")
    assert client_umrs._action_knn(ctx) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["action"] == "knn"
    assert out["bucket"] == "images"
    assert out["path_to_bucket"] == BUCKET
    assert out["results"] == _OK["results"]
    assert out["n_indexed"] == 4


def test_a_server_error_exits_1(image, s3, capsys):
    ctx = _ctx(image, {"status": "error", "message": "No bucket entry found for path: x"},
               output="json")
    assert client_umrs._action_knn(ctx) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "error"
    assert "No bucket entry" in out["message"]
    assert ctx.control_mq.closed


@pytest.mark.parametrize(
    ("alias", "missing_file", "message"),
    [("other", False, "Bucket alias not found"), ("images", True, "File not found")],
)
def test_local_errors_send_nothing(image, s3, capsys, alias, missing_file, message):
    if missing_file:
        image.unlink()
    ctx = _ctx(image, _OK, alias=alias)
    assert client_umrs._action_knn(ctx) == 1
    assert message in capsys.readouterr().out
    assert s3.uploads == []
    assert ctx.control_mq.requests == []
    assert ctx.control_mq.closed


def test_an_upload_failure_is_an_error_line(image, monkeypatch, capsys):
    def _boom(*_a, **_k):
        raise OSError("endpoint unreachable")

    monkeypatch.setattr(client_umrs, "connect_to_s3", _boom)
    ctx = _ctx(image, _OK)
    assert client_umrs._action_knn(ctx) == 1
    assert "endpoint unreachable" in capsys.readouterr().out


def test_knn_parser_and_dispatch():
    parser = argparse.ArgumentParser()
    client_umrs._add_knn_parser(parser.add_subparsers(dest="action"))
    args = parser.parse_args(["knn", "5", "q.png", "images/pets"])
    assert (args.n, args.image_path, args.bucket_alias, args.username) == (5, "q.png",
                                                                           "images/pets", None)
    assert client_umrs._ACTION_HANDLERS["knn"] is client_umrs._action_knn
