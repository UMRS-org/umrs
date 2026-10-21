"""Execution manifest builders (P7a, G-P7-1)."""

from __future__ import annotations

import os
import subprocess

from umrs.jobs.manifest import code_version_ref, environment_snapshot, execution_manifest


class TestEnvironmentSnapshot:
    def test_runtime_facts_present(self):
        snap = environment_snapshot()
        assert snap["python_version"].count(".") >= 2
        assert isinstance(snap["umrs_version"], str) and snap["umrs_version"]
        assert snap["hostname"]
        assert isinstance(snap["gpu_devices"], list)

    def test_container_fields_from_env(self, monkeypatch):
        monkeypatch.setenv("UMRS_CONTAINER_RUNTIME", "singularity")
        monkeypatch.setenv("UMRS_CONTAINER_IMAGE", "ghcr.io/x/mini:v1")
        snap = environment_snapshot()
        assert snap["container_runtime"] == "singularity"
        assert snap["container_image"] == "ghcr.io/x/mini:v1"

    def test_gpu_devices_from_nvidia_visible(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "0,1")
        assert environment_snapshot()["gpu_devices"] == ["0", "1"]

    def test_offline_marker_is_truth_not_invented(self, monkeypatch):
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        assert environment_snapshot()["env_vars"]["vllm_offline"] == ""


class TestCodeVersionRef:
    def test_non_git_dir_reports_empty_not_fabricated(self, tmp_path):
        ref = code_version_ref(str(tmp_path))
        assert ref["commit_sha"] == ""
        assert ref["dirty"] is False

    def test_inherited_git_dir_never_answers_for_repo_root(self, tmp_path, monkeypatch):
        # Under a git hook GIT_DIR names the hook's repository; the manifest
        # must still describe repo_root, not record that repository's SHA.
        decoy = tmp_path / "decoy"
        decoy.mkdir()
        clean = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        git = ["git", "-C", str(decoy), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git, "init", "-q"], env=clean, check=True)
        subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "d"], env=clean, check=True)
        monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
        not_a_repo = tmp_path / "plain"
        not_a_repo.mkdir()
        assert code_version_ref(str(not_a_repo))["commit_sha"] == ""

    def test_repo_root_resolves_a_revision(self):
        # umrs/ is itself a git checkout in CI and developer environments.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if not os.path.isdir(os.path.join(repo_root, ".git")) and not os.path.isfile(
            os.path.join(repo_root, ".git")
        ):
            return  # no checkout in this venue — nothing resolvable to assert
        ref = code_version_ref(repo_root)
        assert len(ref["commit_sha"]) == 40


class TestExecutionManifest:
    def test_shape_and_passthrough(self):
        manifest = execution_manifest(
            provenance={"job_id": "job-1", "status": "completed"},
            trigger="scheduled",
            pipeline_name="demo08",
            tags={"venue": "ci"},
        )
        assert manifest["provenance"]["job_id"] == "job-1"
        assert manifest["trigger"] == "scheduled"
        assert manifest["pipeline_name"] == "demo08"
        assert manifest["tags"] == {"venue": "ci"}
        assert manifest["environment"]["python_version"]
        assert isinstance(manifest["code_versions"], list)
        assert manifest["trace_ids"] == []

    def test_default_trigger_is_api_not_invented(self):
        manifest = execution_manifest(provenance={})
        assert manifest["trigger"] == "api"


# ---------------------------------------------------------------------------
# WP-10 round 2 — model pins into the manifest
# ---------------------------------------------------------------------------


class TestBuildModelPins:
    def test_spec_fields_become_verbatim_pins(self):
        from umrs.jobs.manifest import build_model_pins

        pins = build_model_pins(
            {
                "model": "qwen3-8b",
                "model_version": "v3",
                "model_path": "/models/qwen3-8b",
                "embedding_model": "bge-m3",
            }
        )
        by_name = {p["name"]: p for p in pins}
        assert by_name["qwen3-8b"]["version"] == "v3"
        assert by_name["qwen3-8b"]["path"] == "/models/qwen3-8b"
        assert by_name["qwen3-8b"]["source"] == "spec.model"
        assert by_name["bge-m3"]["source"] == "spec.embedding_model"
        assert "version" not in by_name["bge-m3"]  # never fabricated

    def test_absent_fields_contribute_no_pins(self):
        from umrs.jobs.manifest import build_model_pins

        assert build_model_pins({}) == []
        assert build_model_pins(None) == []
        assert build_model_pins("not-a-dict") == []

    def test_duplicate_names_pin_once_first_source_wins(self):
        from umrs.jobs.manifest import build_model_pins

        pins = build_model_pins({"model": "m1", "model_name": "m1"})
        assert [p["name"] for p in pins] == ["m1"]
        assert pins[0]["source"] == "spec.model"

    def test_manifest_carries_pins_and_records_empty_honestly(self):
        from umrs.jobs.manifest import execution_manifest

        manifest = execution_manifest(provenance={}, model_pins=[])
        assert manifest["model_pins"] == []
        manifest2 = execution_manifest(
            provenance={},
            model_pins=[{"name": "m1", "source": "spec.model"}],
        )
        assert manifest2["model_pins"][0]["name"] == "m1"
