"""D-22/D-21 manifest-contract pins: the hatch.json validator.

The format EXISTS (S8 contract; the fugaku + rikyu-hpc packages carry
the ADR-061 capabilities verbatim) — these pins validate the server-side
validator against the REAL shipped manifests plus the refusal classes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from umrs.mcp.manifest import ManifestError, validate_manifest

# The registry lives in the PARENT checkout (r4-side packages), not umrs/.
_PACKAGES = Path(__file__).resolve().parents[2] / "mcp-servers" / "hatch-registry" / "packages"


def test_real_shipped_manifests_validate():
    """Every REAL hatch.json in the registry validates — no invented shapes."""
    manifests = sorted(_PACKAGES.glob("*/hatch.json"))
    assert manifests, "the registry packages must exist (checked out tree)"
    for path in manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        view = validate_manifest(data)
        assert view["name"] == data["name"]
        assert view["tools"], path.name


def test_fugaku_capabilities_surface():
    """The rich example: per-server egress allowlist + dangerous tools."""
    data = json.loads((_PACKAGES / "fugaku" / "hatch.json").read_text(encoding="utf-8"))
    view = validate_manifest(data)
    caps = view["capabilities"]
    assert caps["transport"] == "stdio"
    assert caps["risk"] == "privileged"
    assert caps["egress_allowlist"] == ["api.fugaku.r-ccs.riken.jp"]
    assert "submit_job" in caps["dangerous_tools"]
    assert set(caps["dangerous_tools"]) <= set(view["tools"])
    assert caps["requires"] == ["fugaku-cert"]


def test_defaults_and_minimal_manifest():
    view = validate_manifest({"name": "x", "version": "0.1.0", "tools": ["t"]})
    caps = view["capabilities"]
    assert caps["transport"] == "stdio"
    assert caps["risk"] == "standard"
    assert caps["egress_allowlist"] == []
    assert caps["readonly_mounts"] == []
    assert caps["dangerous_tools"] == []


@pytest.mark.parametrize(
    "bad,reason",
    [
        ("not-a-dict", "JSON object"),
        ({}, "name"),
        ({"name": "x"}, "version"),
        ({"name": "x", "version": "1"}, "at least one tool"),
        (
            {"name": "x", "version": "1", "tools": ["t"], "capabilities": {"transport": "grpc"}},
            "transport",
        ),
        (
            {"name": "x", "version": "1", "tools": ["t"], "capabilities": {"risk": "wild"}},
            "risk",
        ),
        (
            {
                "name": "x",
                "version": "1",
                "tools": ["t"],
                "capabilities": {"approval": {"dangerous_tools": ["ghost"]}},
            },
            "absent from tools",
        ),
        (
            {
                "name": "x",
                "version": "1",
                "tools": ["t"],
                "capabilities": {"network": "not-an-object"},
            },
            "network must be an object",
        ),
    ],
)
def test_malformed_manifests_refuse_by_name(bad, reason):
    with pytest.raises(ManifestError, match=reason):
        validate_manifest(bad)
