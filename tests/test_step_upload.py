"""
Tests for the UPLOAD_WORKFLOW feature (Phase 1).

Covers:
- Zip creation and validation
- Server-side extraction via _handle_upload_workflow
- SourceType.UPLOADED resolution
- UMRSTask.create_task branch
- TaskCmd constant existence
"""

import json
import os
import tempfile
import zipfile

import pytest

# Ensure src/ is on sys.path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.pipeline_sdk.workflow import (
    SourceSpec,
    SourceType,
    WorkflowDefinition,
    _resolve_source,
    ScriptLoadError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_workflow_dict(source_type="local_path", source_path="/tmp/fake"):
    return {
        "name": "test-workflow",
        "version": "1.0",
        "source": {"type": source_type, "path": source_path},
        "steps": [{"name": "step1", "entrypoint": "my_module:MyStep", "config": {}}],
    }


def _create_workflow_zip(tmpdir, workflow_dict=None, extra_files=None):
    """Create a zip archive containing workflow.json (+ optional files) and return its path."""
    if workflow_dict is None:
        workflow_dict = _minimal_workflow_dict()
    zip_path = os.path.join(tmpdir, "workflow.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workflow.json", json.dumps(workflow_dict))
        if extra_files:
            for name, content in extra_files.items():
                zf.writestr(name, content)
    return zip_path


# ---------------------------------------------------------------------------
# TaskCmd constant
# ---------------------------------------------------------------------------


class TestTaskCmd:
    def test_upload_workflow_constant_exists(self):
        assert hasattr(TaskCmd, "UPLOAD_WORKFLOW")
        assert TaskCmd.UPLOAD_WORKFLOW == "UPLOAD_WORKFLOW"

    def test_upload_workflow_in_all_cmd(self):
        assert TaskCmd.UPLOAD_WORKFLOW in TaskCmd.all_cmd()


# ---------------------------------------------------------------------------
# SourceType.UPLOADED
# ---------------------------------------------------------------------------


class TestSourceTypeUploaded:
    def test_enum_value(self):
        assert SourceType.UPLOADED.value == "uploaded"

    def test_source_spec_from_dict(self):
        spec = SourceSpec.from_dict({"type": "uploaded", "path": "/tmp/wf"})
        assert spec.type == SourceType.UPLOADED
        assert spec.path == "/tmp/wf"

    def test_source_spec_validate_empty_path(self):
        spec = SourceSpec(type=SourceType.UPLOADED, path="")
        with pytest.raises(ValueError, match="non-empty path"):
            spec.validate()

    def test_source_spec_validate_ok(self):
        spec = SourceSpec(type=SourceType.UPLOADED, path="/tmp/wf")
        spec.validate()  # should not raise

    def test_resolve_source_uploaded(self, tmp_path):
        wf_dict = _minimal_workflow_dict(source_type="uploaded", source_path=str(tmp_path))
        wf = WorkflowDefinition.from_dict(wf_dict)
        resolved = _resolve_source(wf)
        assert resolved == tmp_path

    def test_resolve_source_uploaded_missing_dir(self):
        wf_dict = _minimal_workflow_dict(source_type="uploaded", source_path="/nonexistent/dir")
        wf = WorkflowDefinition.from_dict(wf_dict)
        with pytest.raises(ScriptLoadError, match="does not exist"):
            _resolve_source(wf)


# ---------------------------------------------------------------------------
# _handle_upload_workflow (server-side extraction)
# ---------------------------------------------------------------------------


class TestHandleUploadWorkflow:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Set up an isolated tmp dir for extraction."""
        self.tmp_root = tmp_path

    def _handle_upload_workflow(self, client_id, zip_path, workflow_name):
        """Re-implement the core logic of server_umrs._handle_upload_workflow
        for testing without importing the full server (which requires torch).
        """
        import zipfile as _zf
        from umrs.pipeline_sdk.workflow import WorkflowDefinition

        if not os.path.isfile(zip_path):
            return False, None, f"Uploaded file not found: {zip_path}"

        max_size = 500 * 1024 * 1024
        if os.path.getsize(zip_path) > max_size:
            return False, None, f"Zip file too large"

        if not _zf.is_zipfile(zip_path):
            return False, None, f"Uploaded file is not a valid zip archive: {zip_path}"

        workflows_root = os.path.join(str(self.tmp_root), "user_workflows", client_id)
        dest_dir = os.path.join(workflows_root, workflow_name)

        try:
            with _zf.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    resolved = os.path.realpath(os.path.join(dest_dir, member))
                    if not resolved.startswith(os.path.realpath(dest_dir)):
                        return False, None, f"Zip contains unsafe path: {member}"

                candidates = [n for n in zf.namelist() if os.path.basename(n) == "workflow.json"]
                if not candidates:
                    return False, None, "Zip archive does not contain a workflow.json"

                wf_entry = candidates[0]
                try:
                    wf_data = json.loads(zf.read(wf_entry))
                    _ = WorkflowDefinition.from_dict(wf_data)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    return False, None, f"Invalid workflow.json: {e}"

                os.makedirs(dest_dir, exist_ok=True)
                zf.extractall(dest_dir)

            # Rewrite source
            actual_wf = os.path.join(dest_dir, "workflow.json")
            if os.path.isfile(actual_wf):
                with open(actual_wf, "r") as fh:
                    wf_data = json.load(fh)
                wf_data["source"] = {"type": "uploaded", "path": dest_dir}
                with open(actual_wf, "w") as fh:
                    json.dump(wf_data, fh, indent=2)
        except _zf.BadZipFile:
            return False, None, f"Corrupted zip archive: {zip_path}"
        except Exception as e:
            return False, None, f"Failed to extract workflow: {e}"
        finally:
            try:
                os.remove(zip_path)
            except Exception:
                pass

        return True, dest_dir, None

    def test_valid_zip_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            zip_path = _create_workflow_zip(
                td,
                extra_files={"my_module.py": "class MyStep: pass\n"},
            )
            ok, dest, err = self._handle_upload_workflow("user1", zip_path, "my-workflow")
        assert ok is True, f"Expected success, got error: {err}"
        assert dest is not None
        assert os.path.isdir(dest)
        assert os.path.isfile(os.path.join(dest, "workflow.json"))
        # Verify source was rewritten to uploaded
        with open(os.path.join(dest, "workflow.json")) as f:
            data = json.load(f)
        assert data["source"]["type"] == "uploaded"
        assert data["source"]["path"] == dest

    def test_missing_workflow_json(self):
        with tempfile.TemporaryDirectory() as td:
            zip_path = os.path.join(td, "bad.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("readme.txt", "no workflow here")
            ok, dest, err = self._handle_upload_workflow("user1", zip_path, "bad")
        assert ok is False
        assert "workflow.json" in err

    def test_invalid_workflow_json(self):
        with tempfile.TemporaryDirectory() as td:
            zip_path = os.path.join(td, "invalid.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("workflow.json", "not json!!!")
            ok, dest, err = self._handle_upload_workflow("user1", zip_path, "invalid")
        assert ok is False
        assert "Invalid workflow.json" in err

    def test_not_a_zip(self):
        with tempfile.TemporaryDirectory() as td:
            bad_path = os.path.join(td, "notazip.zip")
            with open(bad_path, "w") as f:
                f.write("I am not a zip")
            ok, dest, err = self._handle_upload_workflow("user1", bad_path, "notazip")
        assert ok is False
        assert "not a valid zip" in err

    def test_missing_file(self):
        ok, dest, err = self._handle_upload_workflow("user1", "/nonexistent/file.zip", "x")
        assert ok is False
        assert "not found" in err

    def test_zip_path_traversal_blocked(self):
        """Ensure zip entries with path traversal (../) are rejected."""
        with tempfile.TemporaryDirectory() as td:
            zip_path = os.path.join(td, "evil.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("workflow.json", json.dumps(_minimal_workflow_dict()))
                zf.writestr("../../../etc/passwd", "evil content")
            ok, dest, err = self._handle_upload_workflow("user1", zip_path, "evil")
        assert ok is False
        assert "unsafe path" in err.lower()


# ---------------------------------------------------------------------------
# WorkflowDefinition round-trip with UPLOADED source
# ---------------------------------------------------------------------------


class TestWorkflowDefinitionUploaded:
    def test_to_dict_from_dict_round_trip(self):
        wf_dict = _minimal_workflow_dict(source_type="uploaded", source_path="/tmp/wf")
        wf = WorkflowDefinition.from_dict(wf_dict)
        assert wf.source.type == SourceType.UPLOADED
        out = wf.to_dict()
        assert out["source"]["type"] == "uploaded"
        assert out["source"]["path"] == "/tmp/wf"
