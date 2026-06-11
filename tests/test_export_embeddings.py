"""
Tests for the server-mediated export_embeddings command.

Verifies:
  - _handle_export_embeddings correctly queries the DB and serialises results
  - Error cases: missing bucket, missing table, no embedding column
  - EXPORT_EMBEDDINGS TaskCmd constant exists
  - MessageQueue.request_export_embeddings builds the correct wire message
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Optional

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus


# ---------------------------------------------------------------------------
# Lightweight stand-ins for DB / BucketsCfg (avoid heavy imports)
# ---------------------------------------------------------------------------


@dataclass
class FakeServerBucketEntry:
    bucket_alias: str = "test-bucket"
    s3_username: str = "alice"
    sql_username: str = "sql_alice"
    data_base: Any = None
    db_name: str = "sql_alice_xp1"
    table_name: str = "image_embeddings"
    model: Any = None
    store_embeddings_flag: bool = True
    modality_name: str = "image"
    s3_connection: Any = None
    metadata: Dict = field(default_factory=dict)
    xp_name: str = "xp1"
    num_objects: int = 5
    output_path_to_bucket: Optional[str] = None
    reentrant_lock: RLock = field(default_factory=RLock)


class FakeBucketsCfg:
    def __init__(self, entries=None):
        self._store = entries or {}

    def get(self, key, filter=None):
        entry = self._store.get(key)
        if entry is None or filter is None:
            return entry
        return tuple(getattr(entry, k) for k in filter if hasattr(entry, k))

    def store(self, key, entry):
        self._store[key] = entry

    def items(self):
        return self._store.items()

    def values(self):
        return self._store.values()


class FakeDataBase:
    """Minimal stand-in for umrs.sql.DataBase."""

    def __init__(self, columns, rows):
        self._columns = columns
        self._rows = rows
        self._conn = MagicMock()
        self._cursor = MagicMock()
        # Setup cursor's execute to return column info or data rows
        self._call_count = 0

    def get_conn(self):
        conn = MagicMock()
        cursor = MagicMock()

        def execute_side_effect(query, params=None):
            # First call: column discovery (information_schema)
            # Second call: data fetch (SELECT from table)
            if "information_schema" in str(query):
                cursor.fetchall.return_value = [(c,) for c in self._columns]
            else:
                cursor.fetchall.return_value = self._rows

        cursor.execute = MagicMock(side_effect=execute_side_effect)
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn.cursor = MagicMock(return_value=cursor)
        return conn

    def put_conn(self, conn, close=False):
        pass


# ---------------------------------------------------------------------------
# Tests for TaskCmd
# ---------------------------------------------------------------------------


class TestExportEmbeddingsTaskCmd:
    def test_export_embeddings_constant_exists(self):
        assert hasattr(TaskCmd, "EXPORT_EMBEDDINGS")
        assert TaskCmd.EXPORT_EMBEDDINGS == "EXPORT_EMBEDDINGS"

    def test_export_embeddings_in_all_cmd(self):
        all_cmds = TaskCmd.all_cmd()
        assert "EXPORT_EMBEDDINGS" in all_cmds


# ---------------------------------------------------------------------------
# Tests for _handle_export_embeddings (server-side handler)
# ---------------------------------------------------------------------------


class TestHandleExportEmbeddings:
    """Test the server-side handler function directly."""

    def _import_handler(self):
        """Import the handler; skip if heavy server deps are unavailable."""
        # We patch psycopg2.sql since the handler imports it
        try:
            from server_umrs import _handle_export_embeddings

            return _handle_export_embeddings
        except ImportError:
            pytest.skip("Cannot import server_umrs (missing deps)")

    def test_missing_bucket(self):
        handler = self._import_handler()
        cfg = FakeBucketsCfg({})
        result = json.loads(handler("alice", "nonexistent-bucket;csv", cfg))
        assert result["status"] == "error"
        assert "No bucket entry" in result["message"]

    def test_successful_export_csv(self):
        handler = self._import_handler()
        columns = ["id", "s3_object", "embedding", "model"]
        rows = [
            (1, "bucket/file1.png", "[0.1,0.2,0.3]", "vit"),
            (2, "bucket/file2.png", "[0.4,0.5,0.6]", "vit"),
        ]
        db = FakeDataBase(columns, rows)
        entry = FakeServerBucketEntry(data_base=db)
        cfg = FakeBucketsCfg({"alice.xp1.test-bucket": entry})

        result = json.loads(handler("alice", "alice.xp1.test-bucket;csv", cfg))
        assert result["status"] == "ok"
        assert result["columns"] == columns
        assert result["rows_exported"] == 2
        assert len(result["rows"]) == 2
        assert result["format"] == "csv"
        assert result["database"] == "sql_alice_xp1"
        assert result["table"] == "image_embeddings"

    def test_successful_export_jsonl(self):
        handler = self._import_handler()
        columns = ["id", "s3_object", "embedding"]
        rows = [(1, "b/f.png", "[0.1]")]
        db = FakeDataBase(columns, rows)
        entry = FakeServerBucketEntry(data_base=db)
        cfg = FakeBucketsCfg({"alice.xp1.test-bucket": entry})

        result = json.loads(handler("alice", "alice.xp1.test-bucket;jsonl", cfg))
        assert result["status"] == "ok"
        assert result["format"] == "jsonl"
        assert result["profile"] == "minimal"
        assert result["columns"] == ["s3_object", "embedding"]

    def test_minimal_profile_exports_embedding_with_minimum_metadata(self):
        handler = self._import_handler()
        columns = ["id", "s3_object", "embedding", "model", "created_at"]
        rows = [(1, "bucket/file1.png", "[0.1,0.2]", "vit", "2026-01-01")]
        db = FakeDataBase(columns, rows)
        entry = FakeServerBucketEntry(data_base=db)
        cfg = FakeBucketsCfg({"alice.xp1.test-bucket": entry})

        result = json.loads(handler("alice", "alice.xp1.test-bucket;csv;minimal", cfg))
        assert result["status"] == "ok"
        assert result["profile"] == "minimal"
        assert result["columns"] == ["s3_object", "embedding"]
        assert result["rows"] == [["bucket/file1.png", "[0.1,0.2]"]]

    def test_embedding_only_profile_exports_only_embedding_column(self):
        handler = self._import_handler()
        columns = ["id", "s3_object", "embedding", "model", "created_at"]
        rows = [(1, "bucket/file1.png", "[0.1,0.2]", "vit", "2026-01-01")]
        db = FakeDataBase(columns, rows)
        entry = FakeServerBucketEntry(data_base=db)
        cfg = FakeBucketsCfg({"alice.xp1.test-bucket": entry})

        result = json.loads(handler("alice", "alice.xp1.test-bucket;csv;embedding-only", cfg))
        assert result["status"] == "ok"
        assert result["profile"] == "embedding-only"
        assert result["columns"] == ["embedding"]
        assert result["rows"] == [["[0.1,0.2]"]]

    def test_full_profile_keeps_all_columns(self):
        handler = self._import_handler()
        columns = ["id", "s3_object", "embedding", "model"]
        rows = [(1, "bucket/file1.png", "[0.1,0.2]", "vit")]
        db = FakeDataBase(columns, rows)
        entry = FakeServerBucketEntry(data_base=db)
        cfg = FakeBucketsCfg({"alice.xp1.test-bucket": entry})

        result = json.loads(handler("alice", "alice.xp1.test-bucket;csv;full", cfg))
        assert result["status"] == "ok"
        assert result["profile"] == "full"
        assert result["columns"] == columns

    def test_no_embedding_column(self):
        handler = self._import_handler()
        columns = ["id", "s3_object", "model"]  # no "embedding"
        rows = []
        db = FakeDataBase(columns, rows)
        entry = FakeServerBucketEntry(data_base=db)
        cfg = FakeBucketsCfg({"alice.xp1.test-bucket": entry})

        result = json.loads(handler("alice", "alice.xp1.test-bucket;csv", cfg))
        assert result["status"] == "error"
        assert "no embedding column" in result["message"]

    def test_empty_table(self):
        handler = self._import_handler()
        # Return no columns → "table not found"
        db = FakeDataBase([], [])
        entry = FakeServerBucketEntry(data_base=db)
        cfg = FakeBucketsCfg({"alice.xp1.test-bucket": entry})

        result = json.loads(handler("alice", "alice.xp1.test-bucket;csv", cfg))
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_non_serialisable_values_converted_to_str(self):
        """datetime, date, Decimal etc. should be stringified."""
        import datetime

        handler = self._import_handler()
        columns = ["id", "embedding", "date"]
        rows = [(1, "[0.1]", datetime.date(2025, 1, 15))]
        db = FakeDataBase(columns, rows)
        entry = FakeServerBucketEntry(data_base=db)
        cfg = FakeBucketsCfg({"alice.xp1.test-bucket": entry})

        result = json.loads(handler("alice", "alice.xp1.test-bucket;csv", cfg))
        assert result["status"] == "ok"
        # date should have been converted to string
        assert result["rows"][0][2] == "2025-01-15"

    def test_default_format_is_csv(self):
        handler = self._import_handler()
        columns = ["id", "embedding"]
        rows = [(1, "[0.1]")]
        db = FakeDataBase(columns, rows)
        entry = FakeServerBucketEntry(data_base=db)
        cfg = FakeBucketsCfg({"alice.xp1.test-bucket": entry})

        # No format in args → defaults to csv
        result = json.loads(handler("alice", "alice.xp1.test-bucket", cfg))
        assert result["status"] == "ok"
        assert result["format"] == "csv"
        assert result["profile"] == "minimal"
        assert result["columns"] == ["embedding"]


# ---------------------------------------------------------------------------
# Tests for MessageQueue.request_export_embeddings wire format
# ---------------------------------------------------------------------------


class TestRequestExportEmbeddingsWireFormat:
    """Verify the client method builds the correct ZMQ message."""

    def test_wire_message_format(self):
        """The request string should be EXPORT_EMBEDDINGS:<user>:<path>;<format>;<profile>"""
        mq = MagicMock()
        mq.role = "client"
        mq.socket = MagicMock()
        mq.socket.getsockopt.return_value = 8000

        # Capture what send_string is called with
        sent = []

        def capture_send(msg):
            sent.append(msg)

        mq.send_string = capture_send

        # Mock recv_string to return a valid response
        mq.recv_string = MagicMock(return_value=(None, '{"status":"ok","columns":[],"rows":[]}'))

        # Call the real method on a mock — we just need the wire format logic
        from umrs.core.MessageQueue import MessageQueue

        # Bind the unbound method to our mock
        MessageQueue.request_export_embeddings(
            mq, "alice", "alice.xp1.mybucket", "jsonl", "minimal"
        )

        assert len(sent) == 1
        assert sent[0] == "EXPORT_EMBEDDINGS:alice:alice.xp1.mybucket;jsonl;minimal"

    def test_wire_message_uses_minimal_profile_by_default(self):
        mq = MagicMock()
        mq.role = "client"
        mq.socket = MagicMock()
        mq.socket.getsockopt.return_value = 8000

        sent = []

        def capture_send(msg):
            sent.append(msg)

        mq.send_string = capture_send
        mq.recv_string = MagicMock(return_value=(None, '{"status":"ok","columns":[],"rows":[]}'))

        from umrs.core.MessageQueue import MessageQueue

        MessageQueue.request_export_embeddings(mq, "alice", "alice.xp1.mybucket", "csv")

        assert sent[0] == "EXPORT_EMBEDDINGS:alice:alice.xp1.mybucket;csv;minimal"

    def test_client_only(self):
        """Should raise if called from server role."""
        mq = MagicMock()
        mq.role = "server"
        from umrs.core.MessageQueue import MessageQueue

        with pytest.raises(PermissionError):
            MessageQueue.request_export_embeddings(mq, "alice", "path", "csv")
