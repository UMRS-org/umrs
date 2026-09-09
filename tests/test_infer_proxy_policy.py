"""R1-2: provider path policy + URL join + lease verdicts.

Pins are the rows in design §3.3.3 — every method/prefix class, deny-always
prefixes, opt-in enable/disable, path normalization tricks.
"""

from __future__ import annotations

import pytest
from umrs.infer_proxy.policies import (
    allows,
    inject_credential,
    policy_for,
    strip_credential_headers,
)
from umrs.infer_proxy.urljoin import join_provider_url


class TestOpenAiPolicy:
    policy = policy_for("openai")

    @pytest.mark.parametrize(
        "method,path",
        [
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/responses"),
            ("POST", "/v1/completions"),
            ("POST", "/v1/embeddings"),
            ("POST", "/v1/moderations"),
            ("GET", "/v1/models"),
            ("GET", "/v1/models/gpt-5"),
        ],
    )
    def test_default_open(self, method, path):
        assert allows(self.policy, method, path)

    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/v1/organization/users"),
            ("POST", "/v1/api_keys"),
            ("GET", "/v1/dashboard/billing/subscription"),
            ("GET", "/v1/billing/usage"),
            ("POST", "/v1/realtime"),
            ("POST", "/v1/fine_tuning/jobs"),
            ("DELETE", "/v1/models/x"),
        ],
    )
    def test_default_denied(self, method, path):
        assert not allows(self.policy, method, path)

    def test_opt_in_locked_until_pushed(self):
        assert not allows(self.policy, "POST", "/v1/fine_tuning/jobs", ())
        assert allows(self.policy, "POST", "/v1/fine_tuning/jobs", ("fine_tuning",))
        assert allows(self.policy, "GET", "/v1/files/file-1", ("files",))
        assert not allows(self.policy, "DELETE", "/v1/files/file-1", ("files",))

    def test_normalization_tricks_denied(self):
        assert not allows(self.policy, "GET", "//v1/organization/users")
        assert not allows(self.policy, "GET", "/v1/./organization/users")
        assert not allows(self.policy, "GET", "/V1/ORGANIZATION/users")
        assert not allows(self.policy, "GET", "/v1/../v1/organization")
        assert not allows(self.policy, "GET", "/v1/models/../organization")


class TestAnthropicPolicy:
    policy = policy_for("anthropic")

    def test_default_open(self):
        assert allows(self.policy, "POST", "/v1/messages")
        assert allows(self.policy, "POST", "/v1/messages/count_tokens")
        assert allows(self.policy, "GET", "/v1/models")

    def test_default_denied(self):
        assert not allows(self.policy, "GET", "/v1/organizations/x")
        assert not allows(self.policy, "POST", "/v1/messages/batches", ())
        assert allows(self.policy, "POST", "/v1/messages/batches", ("batches",))

    def test_inject_plain_header(self):
        out = inject_credential(self.policy, {}, "sk-ant-1")
        assert out["x-api-key"] == "sk-ant-1"


class TestAzureAndGoogle:
    def test_azure_whitelist(self):
        policy = policy_for("azure_openai")
        assert allows(policy, "POST", "/openai/deployments/d1/chat/completions")
        assert allows(policy, "POST", "/openai/responses")
        assert allows(policy, "GET", "/openai/models")
        assert not allows(policy, "GET", "/openai/tenants")

    def test_google_whitelist(self):
        policy = policy_for("google")
        assert allows(policy, "POST", "/v1beta/models/gemini:streamGenerateContent")
        assert allows(policy, "GET", "/v1beta/models")
        assert not allows(policy, "POST", "/v1beta/models/gemini:fineTune")
        assert inject_credential(policy, {}, "gkey")["x-goog-api-key"] == "gkey"

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError):
            policy_for("mistral")


class TestStripAndInject:
    def test_strip_all_credential_forms(self):
        inbound = {
            "authorization": "Bearer SESSION-TOKEN",
            "x-api-key": "previous",
            "api-key": "previous",
            "x-goog-api-key": "previous",
            "cookie": "session=abc",
            "openai-organization": "org-1",
            "anthropic-beta": "tools-2024",
        }
        stripped = strip_credential_headers(inbound)
        assert set(stripped) == {"openai-organization", "anthropic-beta"}

    def test_bearer_injection(self):
        out = inject_credential(policy_for("openai"), {}, "sk-real")
        assert out["authorization"] == "Bearer sk-real"


class TestUrlJoin:
    @pytest.mark.parametrize(
        "base,path,expected",
        [
            (
                "https://api.openai.com/v1",
                "/v1/chat/completions",
                "https://api.openai.com/v1/chat/completions",
            ),
            (
                "https://lite.example.org/v1",
                "/chat/completions",
                "https://lite.example.org/v1/chat/completions",
            ),
            ("https://lite.example.org/v1", "/v1/models", "https://lite.example.org/v1/models"),
            ("https://lite.example.org", "/v1/models", "https://lite.example.org/v1/models"),
            (
                "https://x.openai.azure.com/openai",
                "/deployments/d1/chat/completions",
                "https://x.openai.azure.com/openai/deployments/d1/chat/completions",
            ),
            ("https://api.anthropic.com", "/v1/messages", "https://api.anthropic.com/v1/messages"),
            (
                "https://api.openai.com/v1",
                "/v1/chat/completions?stream=true",
                "https://api.openai.com/v1/chat/completions?stream=true",
            ),
        ],
    )
    def test_vectors(self, base, path, expected):
        assert join_provider_url(base, path) == expected
