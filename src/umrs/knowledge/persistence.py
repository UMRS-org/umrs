"""Local persistence helpers for Phase 4 ``ExtractedKnowledge``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .compiled_skill import CompiledSkill, SkillReview
from .ir import KnowledgeIR

logger = logging.getLogger(__name__)


def extracted_knowledge_dir(storage_path: str | Path) -> Path:
    """Directory for per-artifact structured knowledge JSON files."""
    return Path(storage_path) / "extracted_knowledge"


def _artifact_filename(artifact_id: str) -> str:
    return f"{quote(artifact_id, safe='')}.json"


def write_local_extracted_knowledge(
    persistence_dir: str | Path,
    artifact_id: str,
    knowledge: dict[str, Any],
) -> Path:
    """Persist structured knowledge to disk; return the file path."""
    directory = Path(persistence_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _artifact_filename(artifact_id)
    path.write_text(
        json.dumps(knowledge, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def read_local_extracted_knowledge(
    persistence_dir: str | Path,
    artifact_id: str,
) -> dict[str, Any] | None:
    """Load structured knowledge from disk, or None if missing."""
    path = Path(persistence_dir) / _artifact_filename(artifact_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read local extracted knowledge %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


# ── KnowledgeIR (canonical persistence record) ───────────────────────────────


def knowledge_ir_dir(storage_path: str | Path) -> Path:
    """Directory for canonical per-artifact ``KnowledgeIR`` JSON files."""
    return Path(storage_path) / "knowledge_ir"


def write_local_knowledge_ir(
    ir_dir: str | Path,
    ir: KnowledgeIR,
) -> Path:
    """Persist a ``KnowledgeIR`` write-once; return the file path.

    Write-once: if a record for this ``ir_id`` already exists (same content and
    extractor version), it is left untouched — the record is canonical and
    immutable.
    """
    directory = Path(ir_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ir.ir_id}.json"
    if path.is_file():
        return path
    path.write_text(
        json.dumps(ir.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def read_local_knowledge_ir(
    ir_dir: str | Path,
    ir_id: str,
) -> KnowledgeIR | None:
    """Load a ``KnowledgeIR`` from disk, or None if missing or unreadable."""
    path = Path(ir_dir) / f"{ir_id}.json"
    if not path.is_file():
        return None
    return _load_knowledge_ir_file(path)


def read_all_local_knowledge_ir(ir_dir: str | Path) -> list[KnowledgeIR]:
    """Load every canonical ``KnowledgeIR`` under *ir_dir* (malformed skipped).

    This is the regeneration input: derived indexes (Phase 1 sinks) are rebuilt
    solely from these canonical records.
    """
    directory = Path(ir_dir)
    if not directory.is_dir():
        return []
    records: list[KnowledgeIR] = []
    for path in sorted(directory.glob("*.json")):
        ir = _load_knowledge_ir_file(path)
        if ir is not None:
            records.append(ir)
    return records


def _load_knowledge_ir_file(path: Path) -> KnowledgeIR | None:
    """Load and validate a single ``KnowledgeIR`` JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read local knowledge IR %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return KnowledgeIR.from_dict(payload)
    except (KeyError, TypeError) as exc:
        logger.warning("Malformed knowledge IR %s: %s", path, exc)
        return None


# ── Derived indexes (Phase 1 — regenerable from KnowledgeIR) ─────────────────


def knowledge_graph_index_path(storage_path: str | Path) -> Path:
    """Path of the canonical local knowledge-graph derived index."""
    return Path(storage_path) / "knowledge_graph.json"


def write_local_knowledge_graph(
    storage_path: str | Path,
    graph: dict[str, Any],
) -> Path:
    """Persist the derived knowledge-graph index (overwrite-latest).

    Unlike ``KnowledgeIR`` (immutable, write-once), the graph is a *derived*
    index: it is rewritten whenever it is recompiled from the canonical IRs
    (e.g. after the linker improves).
    """
    path = knowledge_graph_index_path(storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(graph, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def read_local_knowledge_graph(storage_path: str | Path) -> dict[str, Any] | None:
    """Load the derived knowledge-graph index, or None if missing/unreadable."""
    path = knowledge_graph_index_path(storage_path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read local knowledge graph %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


# ── Compiled skills (Phase 3 — human-reviewed continuous learning) ───────────


def compiled_skill_dir(storage_path: str | Path) -> Path:
    """Directory for compiled procedural skills (write-once, pending review)."""
    return Path(storage_path) / "compiled_skills"


def write_local_compiled_skill(
    storage_path: str | Path,
    skill: CompiledSkill,
) -> Path:
    """Persist a compiled skill write-once; return the file path.

    Write-once: ``skill_id`` is a deterministic hash of the source IR lineage,
    so recompiling the same understanding never overwrites or duplicates a skill.
    Human review decisions are stored separately (see ``write_local_skill_reviews``)
    and never mutate the compiled skill itself.
    """
    directory = compiled_skill_dir(storage_path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{skill.skill_id}.json"
    if path.is_file():
        return path
    path.write_text(
        json.dumps(skill.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def read_all_local_compiled_skills(storage_path: str | Path) -> list[CompiledSkill]:
    """Load every compiled skill under *storage_path* (malformed skipped)."""
    directory = compiled_skill_dir(storage_path)
    if not directory.is_dir():
        return []
    skills: list[CompiledSkill] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            skills.append(CompiledSkill.from_dict(payload))
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
            logger.warning("Failed to read compiled skill %s: %s", path, exc)
    return skills


def skill_reviews_path(storage_path: str | Path) -> Path:
    """Path of the human-review decisions index (overwrite-latest)."""
    return Path(storage_path) / "skill_reviews.json"


def write_local_skill_reviews(
    storage_path: str | Path,
    reviews: dict[str, SkillReview],
) -> Path:
    """Persist the human-review decisions index (overwrite-latest).

    Reviews are mutable, latest-wins state — distinct from the immutable
    compiled skills they annotate.
    """
    path = skill_reviews_path(storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {sid: review.to_dict() for sid, review in sorted(reviews.items())}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def read_local_skill_reviews(storage_path: str | Path) -> dict[str, SkillReview]:
    """Load the human-review decisions index, or empty if missing/unreadable."""
    path = skill_reviews_path(storage_path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read skill reviews %s: %s", path, exc)
        return {}
    if not isinstance(payload, dict):
        return {}
    reviews: dict[str, SkillReview] = {}
    for sid, entry in payload.items():
        if isinstance(entry, dict):
            try:
                reviews[sid] = SkillReview.from_dict(entry)
            except (KeyError, TypeError):
                continue
    return reviews
