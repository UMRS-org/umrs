"""Static guards for production deployment configuration."""

from pathlib import Path
import re


UMRS_ROOT = Path(__file__).resolve().parents[1]
RCCS_CLOUD_SITE_CONFIG = UMRS_ROOT / "deploy" / "rccs-cloud" / "site.yaml"


FORBIDDEN_RCCS_SITE_PATTERNS = {
    r"umrs-rccs-litellm-key": "committed LiteLLM service key",
    r"umrs_rccs": "committed PostgreSQL password",
    r"/mnagaso(?:/|$)": "operator-specific home directory",
    r"registry_dsn:\s*postgres(?:ql)?://[^:\s]+:[^@\s]+@": "password-bearing registry DSN",
}


def test_rccs_cloud_site_config_has_no_committed_operator_secrets():
    text = RCCS_CLOUD_SITE_CONFIG.read_text()

    violations = [
        description
        for pattern, description in FORBIDDEN_RCCS_SITE_PATTERNS.items()
        if re.search(pattern, text)
    ]

    assert not violations, "Forbidden RCCS site config literals: " + ", ".join(violations)
