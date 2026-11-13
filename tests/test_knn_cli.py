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
        self.deleted = []

    def upload_file(self, path, bucket, key):
        self.uploads.append((path, bucket, key))

    def delete_object(self, Bucket, Key):  # noqa: N803 — boto3 signature
        self.deleted.append((Bucket, Key))


class _MQ:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        self.closed = False

    def request_knn(self, client_id, n, s3_object, bucket, subdir=None):
        self.requests.append((client_id, n, s3_object, bucket, subdir))
        if isinstance(self.reply, BaseException):
            raise self.reply
        if self.reply is None or isinstance(self.reply, str):
            return self.reply  # raw, non-JSON server replies
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
    fake.connects = []

    def _connect(*args, **kwargs):
        fake.connects.append((args, kwargs))
        return fake, None

    monkeypatch.setattr(client_umrs, "connect_to_s3", _connect)
    return fake


def _ctx(image, reply, alias="images", output="text", username=None, connection=None,
         s3cfg="/nonexistent/s3cfg"):
    args = argparse.Namespace(n=2, image_path=str(image), bucket_alias=alias,
                              username=username, output=output, verbose=False)
    entry = SimpleNamespace(path_to_bucket=BUCKET, s3_configuration_file=s3cfg)
    return SimpleNamespace(args=args, control_mq=_MQ(reply), connection=connection,
                           user={"username": "alice"}, buckets_cfg=_Cfg({"images": entry}))


# --- which s3cfg and endpoint the query upload uses ---------------------------------


def test_the_bucket_s3cfg_is_used_with_no_endpoint_override(image, s3, tmp_path):
    cfg = tmp_path / "bucket.s3cfg"
    cfg.write_text("[default]\n", encoding="utf-8")
    assert client_umrs._action_knn(_ctx(image, _OK, s3cfg=str(cfg))) == 0
    assert s3.connects == [((str(cfg),), {"endpoint_url_override": None})]


def test_a_missing_s3cfg_falls_back_to_the_container_default(image, s3, tmp_path, monkeypatch):
    default = tmp_path / "docker.s3cfg"
    default.write_text("[default]\n", encoding="utf-8")
    monkeypatch.setattr(client_umrs, "get_default_docker_s3cfg", lambda: str(default))
    assert client_umrs._action_knn(_ctx(image, _OK)) == 0
    assert s3.connects == [((str(default),), {"endpoint_url_override": None})]


def test_a_profile_overrides_the_endpoint(image, s3):
    profile = SimpleNamespace(profile_name="front", minio_url="http://front:9000")
    assert client_umrs._action_knn(_ctx(image, _OK, connection=profile)) == 0
    assert s3.connects[0][1] == {"endpoint_url_override": "http://front:9000"}


def test_a_connection_without_profile_keeps_the_s3cfg_endpoint(image, s3):
    unnamed = SimpleNamespace(profile_name=None, minio_url="http://front:9000")
    assert client_umrs._action_knn(_ctx(image, _OK, connection=unnamed)) == 0
    assert s3.connects[0][1] == {"endpoint_url_override": None}


# --- replies that are not a JSON object ------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "message"),
    [("SERVER_OVERLOADED", "Server overloaded"),
     ("UNAUTHORIZED:KNN", "Not authorized"),
     (TimeoutError("Timeout waiting for KNN reply"), "Timeout waiting for KNN reply"),
     (None, "No reply")],
)
def test_non_json_replies_are_clear_error_lines(image, s3, capsys, reply, message):
    """Review U1: a timeout surfaced as 'the JSON object must be str ... not
    NoneType' and SERVER_OVERLOADED as 'Expecting value: line 1 column 1'."""
    ctx = _ctx(image, reply, output="json")
    assert client_umrs._action_knn(ctx) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "error"
    assert message in out["message"]
    assert ctx.control_mq.closed


def test_subdir_alias_uploads_under_it_and_scopes_the_search(image, s3, capsys):
    ctx = _ctx(image, _OK, alias="images/pets")

    assert client_umrs._action_knn(ctx) == 0

    ((path, bucket, key),) = s3.uploads
    assert (path, bucket) == (str(image), BUCKET)
    assert key.startswith("pets/.umrs-knn/")
    assert key.endswith("-q.png")
    assert ctx.control_mq.requests == [("alice", 2, key, BUCKET, "pets")]
    assert capsys.readouterr().out.strip() == _OK["summary"]
    assert ctx.control_mq.closed


def test_the_query_never_overwrites_an_object_of_the_bucket(image, s3):
    """Review U1: the query went up under its basename, so `umrs knn 5
    img001.jpg` replaced the bucket's own (maybe a collaborator's) img001.jpg."""
    ctx = _ctx(image, _OK, username="alice")
    statuses = [client_umrs._action_knn(ctx),
                client_umrs._action_knn(_ctx(image, _OK, username="alice"))]

    keys = [key for _path, _bucket, key in s3.uploads]
    shapes = [(key.split("/")[0], key.rsplit("-", 1)[1]) for key in keys]
    assert (statuses, shapes) == ([0, 0], [(".umrs-knn", "q.png")] * 2)
    assert len(set(keys)) == 2  # one fresh key per query
    assert ctx.control_mq.requests == [("alice", 2, keys[0], BUCKET, None)]
    assert s3.deleted == [(BUCKET, key) for key in keys]  # removed after the reply


def test_the_query_object_is_removed_after_an_error_reply(image, s3):
    ctx = _ctx(image, {"status": "error", "message": "no"})
    assert client_umrs._action_knn(ctx) == 1
    ((_path, _bucket, key),) = s3.uploads
    assert s3.deleted == [(BUCKET, key)]


def test_the_reserved_query_dir_matches_the_server():
    from umrs.server.knn_worker import QUERY_UPLOAD_DIR

    assert client_umrs._KNN_QUERY_DIR == QUERY_UPLOAD_DIR


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
