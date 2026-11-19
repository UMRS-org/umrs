"""Static D-17 census of object-store consumers under ``src/umrs``.

The census is the hand-off boundary for later bound-store work, not a runtime
migration or a claim that the knowledge plane follows ADR-077.  Module paths
are relative to ``src/umrs``.  ``follows_binding`` entries name the existing
resolver they use; every other matching module stays explicitly classified so
new consumers cannot appear unnoticed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

UMRS_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = UMRS_ROOT / "src" / "umrs"
STORE_MARKERS = frozenset({"get_s3_config", "_build_s3_client", "S3_ENDPOINT_URL"})
DECLARED_STORE_CONSUMER_CLASSES = frozenset(
    {
        "config_definition",
        "follows_binding",
        "server_default",
        "worker_default",
        "worker_environment",
    }
)
DECLARED_BINDING_RESOLVERS = frozenset({"project_s3_config", "resolve_bucket_store"})

STORE_CONSUMER_CENSUS = {
    # ADR-077/115 fail-closed followers, including the D17-b ingestion fix.
    "api/buckets.py": "follows_binding",
    "api/files.py": "follows_binding",
    "jobs/data_ingestion.py": "follows_binding",
    "jobs/workflow_run.py": "follows_binding",
    # The eight knowledge-plane modules remain server-default (D17-d not started).
    "api/bundle_restore.py": "server_default",
    "api/knowledge.py": "server_default",
    "api/knowledge_artifact_delete.py": "server_default",
    "api/knowledge_upload.py": "server_default",
    "api/knowledge_v2.py": "server_default",
    "api/rag_spine.py": "server_default",
    "jobs/knowledge_adapters.py": "server_default",
    "knowledge/retrieval/backfill.py": "server_default",
    # Other server-side readers and maintenance commands use the default store.
    "api/catalog.py": "server_default",
    "api/intake_uploads.py": "server_default",
    "api/jobs.py": "server_default",
    "api/projects.py": "server_default",
    "api/vision.py": "server_default",
    "data_catalog/backfill.py": "server_default",
    "data_catalog/reconcile.py": "server_default",
    "jobs/batch_infer.py": "server_default",
    "jobs/model_eval.py": "server_default",
    "plugins/runner.py": "server_default",
    # The shared definition itself matches the environment marker.
    "shared/storage/s3_config.py": "config_definition",
    # These workers construct their own clients from injected process settings.
    "plugins/plugin_runner/entrypoint.py": "worker_default",
    "server/step_worker.py": "worker_default",
    "vision_service/service.py": "worker_default",
    # These launch/renewal paths forward endpoint settings; they build no client.
    "mini-rivault-llm/sts_renewal.py": "worker_environment",
    "orchestrator/slurm_vision_script.py": "worker_environment",
    "server/slurm_step_executor.py": "worker_environment",
    "server/step_executor.py": "worker_environment",
}

BINDING_FOLLOWER_RESOLVERS = {
    "api/buckets.py": "resolve_bucket_store",
    "api/files.py": "resolve_bucket_store",
    "jobs/data_ingestion.py": "resolve_bucket_store",
    "jobs/workflow_run.py": "project_s3_config",
}


def _constant_symbol(value: object) -> str:
    if not isinstance(value, str):
        return ""
    markers = (
        marker for marker in STORE_MARKERS if value == marker or value.startswith(f"{marker}=")
    )
    return next(markers, "")


def _node_symbol(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.Constant):
        return _constant_symbol(node.value)
    return ""


def _matching_modules() -> dict[str, set[str]]:
    matches: dict[str, set[str]] = {}
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = {
            symbol for node in ast.walk(tree) if (symbol := _node_symbol(node)) in STORE_MARKERS
        }
        if found:
            matches[path.relative_to(SOURCE_ROOT).as_posix()] = found
    return matches


def test_no_unclassified_store_consumer() -> None:
    unclassified = sorted(_matching_modules().keys() - STORE_CONSUMER_CENSUS.keys())
    if unclassified:
        pytest.fail(f"classify every store consumer in the D-17 census: {unclassified}")


def test_no_classified_module_is_stale() -> None:
    stale = sorted(STORE_CONSUMER_CENSUS.keys() - _matching_modules().keys())
    if stale:
        pytest.fail(f"remove or reclassify census rows that no longer match: {stale}")


def test_every_class_belongs_to_the_declared_set() -> None:
    undeclared = sorted(set(STORE_CONSUMER_CENSUS.values()) - DECLARED_STORE_CONSUMER_CLASSES)
    if undeclared:
        pytest.fail(f"declare a class before using it in the census: {undeclared}")


def test_every_binding_follower_names_its_resolver() -> None:
    followers = {path for path, kind in STORE_CONSUMER_CENSUS.items() if kind == "follows_binding"}
    resolvers = set(BINDING_FOLLOWER_RESOLVERS.values())
    if followers != BINDING_FOLLOWER_RESOLVERS.keys():
        pytest.fail("every follows_binding row must name exactly one resolver")
    if not resolvers <= DECLARED_BINDING_RESOLVERS:
        pytest.fail(f"undeclared resolver names: {sorted(resolvers - DECLARED_BINDING_RESOLVERS)}")
    if not all(resolvers):
        pytest.fail("resolver names cannot be empty")
