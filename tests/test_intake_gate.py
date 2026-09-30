"""I2 IntakeGate tests — admission rules, secret scan, runner integration."""

from __future__ import annotations

from umrs.connectors import (
    ConnectorConfig,
    ConnectorType,
    Document,
    DocumentStatus,
)
from umrs.connectors.intake_gate import IntakeGate, IntakePolicy
from umrs.connectors.sync_runner import ConnectorSyncRunner
from umrs.connectors.sync_state import InMemorySyncStateStore

# Interface-parity signatures: fakes intentionally ignore args.


def _doc(source_id, content, content_type="text/plain"):
    return Document(
        source_id=source_id,
        content=content,
        content_type=content_type,
        status=DocumentStatus.PENDING,
    )


class TestIntakeGateRules:
    def test_allows_plain_text(self):
        gate = IntakeGate()
        decision = gate.inspect(_doc("paper.md", "knowledge"))
        assert decision.action == "allow"

    def test_blocks_aws_access_key(self):
        gate = IntakeGate()
        assert gate.inspect(_doc("cfg.txt", "AKIAIOSFODNN7EXAMPLE")).action == "block"

    def test_blocks_p_pem_private_key(self):
        gate = IntakeGate()
        dec = gate.inspect(_doc("id.pem", "-----BEGIN PRIVATE KEY-----\nXYZ"))
        assert dec.action == "block" and dec.pattern == "private_key_pem"

    def test_blocks_github_token(self):
        gate = IntakeGate()
        token = "ghp_" + "abcdef0123456789" * 3  # len > 36
        dec = gate.inspect(_doc("env.txt", f"TOKEN={token}"))
        assert dec.action == "block" and dec.pattern == "github_token"

    def test_blocks_api_key_assignment(self):
        gate = IntakeGate()
        dec = gate.inspect(_doc("conf.json", '{"api_key": "xxxxYYYYzzzzAAAAaaaa"}'))
        assert dec.action == "block" and dec.pattern == "api_key_assignment"

    def test_blocks_signed_jwt(self):
        gate = IntakeGate()
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            + "."
            + "eyJzdWIiOiIxMjM0NTY3ODkwIn0"
            + "."
            + "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        dec = gate.inspect(_doc("note.txt", jwt))
        assert dec.action == "block" and dec.pattern == "generic_jwt"

    def test_blocks_password_assignment(self):
        gate = IntakeGate()
        dec = gate.inspect(_doc("env.env", "DB_PASSWORD=supersecret1"))
        assert dec.action == "block" and dec.pattern == "password_assignment"

    def test_docs_run_cap(self):
        gate = IntakeGate(IntakePolicy(max_docs_per_run=1))
        dec = gate.inspect(_doc("a.txt", "ok"), doc_index=0)
        assert dec.action == "allow"
        dec = gate.inspect(_doc("b.txt", "ok"), doc_index=1)
        assert dec.action == "block" and dec.pattern == "max_docs_per_run"

    def test_content_size_cap(self):
        gate = IntakeGate(IntakePolicy(max_content_bytes=8))
        dec = gate.inspect(_doc("big.txt", "0123456789abcdef"))
        assert dec.action == "block" and dec.pattern == "max_content_bytes"

    def test_supported_format_passes(self):
        gate = IntakeGate()
        dec = gate.inspect(_doc("visual.png", "data", content_type="image/png"))
        assert dec.action == "block" and dec.pattern == "unsupported_format"


class TestRunnerGateIntegration:
    def _runner(self):
        return ConnectorSyncRunner(InMemorySyncStateStore())

    def _config(self):
        return ConnectorConfig(connector_type=ConnectorType.CUSTOM, name="demo")

    def test_secrets_never_ingested(self):
        ingested = []
        runner = self._runner()
        report = runner.run(
            _DummyConnector(),
            self._config(),
            ingest=ingested.append,
            docs_iter=iter(
                [
                    _doc("good.txt", "clean content"),
                    _doc("evil.txt", "AKIAIOSFODNN7EXAMPLE"),
                ]
            ),
        )
        assert len(ingested) == 1
        assert ingested[0].source_id == "good.txt"
        assert report.docs_blocked == 1
        assert report.scan_patterns.get("aws_access_key") == 1

    def test_dlq_records_blocked_docs(self):
        runner = self._runner()
        runner.run(
            _DummyConnector(),
            self._config(),
            docs_iter=iter([_doc("secret.pem", "-----BEGIN PRIVATE KEY-----\nx")]),
        )
        state = runner._store.get("demo")
        assert state.last_status == "ok"
        assert any(d.get("blocked_by") == "private_key_pem" for d in state.dlq)
        assert "last_scan" in state.checkpoint_json
        assert state.checkpoint_json["last_scan"]["blocked"] == 1
        assert state.checkpoint_json["last_scan"]["patterns"].get("private_key_pem") == 1

    def test_secrets_never_ingested_repeat_runs_dedup_ram_store(self):
        runner = self._runner()
        docs = [_doc("secret.txt", "AKIAIOSFODNN7EXAMPLE"), _doc("ok.txt", "ok")]
        for _ in range(2):
            report = runner.run(_DummyConnector(), self._config(), docs_iter=iter(docs))
        state = runner._store.get("demo")
        assert state.docs_synced == 1
        assert state.errors_count == 0
        assert report.docs_blocked == 1

    def test_custom_policy_extends_defaults(self):
        pol = IntakePolicy(custom_patterns=(("project_secret", r"PROJSECRET\d{6}"),))
        gate = IntakeGate(pol)
        dec = gate.inspect(_doc("x", "value PROJSECRET012345"))
        assert dec.action == "block" and dec.pattern == "project_secret"


class _DummyConnector:
    name = "dummy"

    @property
    def connector_type(self):
        return ConnectorType.CUSTOM

    def validate_config(self, config):
        return []

    def fetch(self, config, checkpoint=None):
        return iter([])
