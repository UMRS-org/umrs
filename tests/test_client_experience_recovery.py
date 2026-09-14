from __future__ import annotations

import json
from argparse import Namespace

# splice removed (ADR-053): umrs installed editable in .venv
import client_umrs


def test_parse_recoverable_bucket_name_handles_output_bucket():
    parsed = client_umrs._parse_recoverable_bucket_name(
        "user1.image-indexing.image-input-output",
        "user1",
    )

    assert parsed is not None
    assert parsed.xp_name == "image-indexing"
    assert parsed.bucket_alias == "image-input"
    assert parsed.output is True


def test_build_recovered_experience_uses_pgvector_status_and_output_bucket():
    buckets = [
        client_umrs._RecoveredBucket(
            bucket="user1.image-indexing.image-input",
            username="user1",
            xp_name="image-indexing",
            bucket_alias="image-input",
        ),
        client_umrs._RecoveredBucket(
            bucket="user1.image-indexing.image-input-output",
            username="user1",
            xp_name="image-indexing",
            bucket_alias="image-input",
            output=True,
        ),
    ]
    tables = [
        client_umrs._RecoveredTable(
            db_name="image_indexing_usr1_img",
            xp_name="image_indexing",
            sql_username="usr1_img",
            table_name="image_input",
            columns=[
                {"name": "id", "type": "integer"},
                {"name": "s3_object", "type": "text"},
                {"name": "embedding", "type": "vector(1024)"},
                {"name": "sample_id", "type": "text"},
            ],
            vector_size=1024,
        )
    ]

    recovered = client_umrs._build_recovered_experience_fragments(
        username="user1",
        buckets=buckets,
        tables=tables,
        model_name="sam3",
        model_type="TransformerEncoder",
        fallback_vector_size="768",
        return_type="auto",
    )

    assert recovered == [
        {
            "xp_name": "image_indexing",
            "sql_username": "usr1_img",
            "modalities": [
                {
                    "modality_name": "image_input",
                    "s3": {
                        "bucket_alias": "image-input",
                        "data": {
                            "type": "file",
                            "mime_type": "application/octet-stream",
                        },
                        "need_output_bucket": True,
                    },
                    "model": {
                        "model_name": "sam3",
                        "vector_size": "1024",
                        "model_type": "TransformerEncoder",
                        "return_type": "file",
                    },
                    "sql": {
                        "store_embeddings": "yes",
                        "table_name": "image_input",
                        "meta_data": [{"name": "sample_id", "type": "TEXT"}],
                    },
                }
            ],
        }
    ]


def test_experience_recover_writes_json_without_live_services(tmp_path, monkeypatch, capsys):
    out_path = tmp_path / "recovered.json"
    bucket = client_umrs._RecoveredBucket(
        bucket="user1.demo.text",
        username="user1",
        xp_name="demo",
        bucket_alias="text",
        object_count=3,
        total_size_bytes=123,
    )
    table = client_umrs._RecoveredTable(
        db_name="demo_usr1",
        xp_name="demo",
        sql_username="usr1",
        table_name="text",
        columns=[
            {"name": "s3_object", "type": "text"},
            {"name": "embedding", "type": "vector(768)"},
        ],
        vector_size=768,
    )
    monkeypatch.setattr(
        client_umrs, "_list_recovery_buckets_from_s3", lambda **kwargs: ([bucket], [])
    )
    monkeypatch.setattr(
        client_umrs, "_list_recovery_postgres_status", lambda xp_names: ([table], [])
    )

    args = Namespace(
        experience_action="recover",
        xp_names=[],
        username="user1",
        conf=None,
        s3cfg=str(tmp_path / "s3cfg"),
        output_path=str(out_path),
        merge=False,
        update=False,
        no_postgres=False,
        no_object_count=False,
        model_name="intfloat/e5-base-v2",
        model_type="TransformerEncoder",
        vector_size="768",
        return_type="auto",
        output="json",
    )

    assert client_umrs._handle_experience_command(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["experiences"][0]["xp_name"] == "demo"
    assert payload["s3_buckets"][0]["object_count"] == 3
    assert json.loads(out_path.read_text(encoding="utf-8"))["xp_name"] == "demo"


def test_experience_remove_alias_edits_user_json(tmp_path, capsys):
    user_cfg = tmp_path / "user1.json"
    user_cfg.write_text(
        json.dumps(
            {
                "username": "user1",
                "fullname": "User One",
                "s3_configuration_file": str(tmp_path / "s3cfg"),
                "experiences": [
                    {"xp_name": "keep", "sql_username": "usr1", "modalities": []},
                    {"xp_name": "drop", "sql_username": "usr1", "modalities": []},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    args = Namespace(
        experience_action="remove",
        xp_names=["drop"],
        conf=str(user_cfg),
        output="json",
    )

    assert client_umrs._handle_experience_command(args) == 0

    payload = json.loads(capsys.readouterr().out)
    updated = json.loads(user_cfg.read_text(encoding="utf-8"))
    assert payload["removed"] == ["drop"]
    assert payload["resources_retained"] is True
    assert [xp["xp_name"] for xp in updated["experiences"]] == ["keep"]


def test_experience_list_outputs_experiences(tmp_path, capsys):
    user_cfg = tmp_path / "user1.json"
    user_cfg.write_text(
        json.dumps(
            {
                "username": "user1",
                "fullname": "User One",
                "s3_configuration_file": str(tmp_path / "s3cfg"),
                "experiences": [
                    {
                        "xp_name": "image-indexing",
                        "sql_username": "usr1_img",
                        "modalities": [
                            {
                                "modality_name": "image_input",
                                "s3": {
                                    "bucket_alias": "image-input",
                                    "data": {
                                        "type": "file",
                                        "mime_type": "application/octet-stream",
                                    },
                                    "need_output_bucket": True,
                                },
                                "model": {
                                    "model_name": "sam3",
                                    "vector_size": "1024",
                                    "model_type": "TransformerEncoder",
                                    "return_type": "file",
                                },
                                "sql": {
                                    "store_embeddings": "yes",
                                    "table_name": "image_input",
                                    "meta_data": [{"name": "sample_id", "type": "TEXT"}],
                                },
                            }
                        ],
                    },
                    {
                        "xp_name": "text-search",
                        "sql_username": "usr1_txt",
                        "modalities": [],
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    args = Namespace(
        experience_action="list",
        conf=str(user_cfg),
        output="json",
    )

    assert client_umrs._handle_experience_command(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["count"] == 2
    assert payload["experiences"][0]["xp_name"] == "image-indexing"
    assert payload["experiences"][1]["xp_name"] == "text-search"
    assert payload["target"] == str(user_cfg)
