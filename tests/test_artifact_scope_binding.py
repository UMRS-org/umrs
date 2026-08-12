"""Artifact-route caller-scope binding (M10 MED-HIGH).

Pre-fix every artifact route was anonymous: project filter honored
verbatim, mutations executed by UUID alone — any sidecar-reachable caller
could enumerate/read/overwrite/delete any project's artifacts (same
pattern family as '''wave-1 artifact-routes conn-pool fix post-state''').
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from umrs.api import artifact_routes as ar


def _art(project_id):
    from umrs.artifact_registry.models import Artifact

    return Artifact(
        id=None,
        artifact_type="file",
        name="x",
        owner_id=None,
        project_id=project_id,
        uri="s3://b/k",
        bucket="b",
        s3_key="k",
        size_bytes=1,
        checksum=None,
        metadata={},
        execution_id=None,
        workflow_id=None,
        job_id=None,
        version=1,
    )


class TestScopeCheckedProject:
    def test_caller_project_forced_for_non_admin(self):
        assert ar._scope_checked_project("proj-a", False, "evil-proj") == "proj-a"

    def test_requested_filter_honored_when_admin(self):
        assert ar._scope_checked_project("proj-a", True, "other-proj") == "other-proj"

    def test_admin_may_unscope(self):
        assert ar._scope_checked_project("proj-a", True, None) is None


class TestArtifactScope:
    def test_matching_project_allows(self):
        ar._require_artifact_scope(_art("proj-a"), "proj-a", False)

    def test_other_project_denied_as_404(self):
        with pytest.raises(HTTPException) as exc:
            ar._require_artifact_scope(_art("proj-b"), "proj-a", False)
        assert exc.value.status_code == 404

    def test_admin_permitted_everywhere(self):
        ar._require_artifact_scope(_art("proj-b"), "proj-a", True)

    def test_none_artifact_forces_404(self):
        with pytest.raises(HTTPException) as exc:
            ar._require_artifact_scope(None, "proj-a", False)
        assert exc.value.status_code == 404
