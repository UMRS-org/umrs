"""UMRS Knowledge Graph module.

Builds entity/relation tables from indexed documents to support
graph-augmented retrieval. Runs as a background worker that processes
newly indexed documents and extracts entities and relations.

No dependency on r4.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Data models ─────────────────────────────────────────────────────────────


class EntityType(Enum):
    """Common entity types extracted from documents."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    SOFTWARE = "software"
    DATASET = "dataset"
    METHOD = "method"
    METRIC = "metric"
    PUBLICATION = "publication"
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Common relation types between entities."""

    AUTHORED_BY = "authored_by"
    USES = "uses"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    DEPENDS_ON = "depends_on"
    CITES = "cites"
    IMPLEMENTS = "implements"
    PRODUCES = "produces"
    EVALUATES = "evaluates"
    COMPARED_WITH = "compared_with"


@dataclass
class Entity:
    """An entity extracted from a document."""

    entity_id: str = ""
    name: str = ""
    entity_type: EntityType = EntityType.UNKNOWN
    aliases: list[str] = field(default_factory=list)
    source_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    mention_count: int = 1

    def __post_init__(self) -> None:
        if not self.entity_id:
            self.entity_id = hashlib.sha256(
                f"{self.entity_type.value}:{self.name.lower()}".encode()
            ).hexdigest()[:16]


@dataclass
class Relation:
    """A relation between two entities."""

    source_entity_id: str = ""
    target_entity_id: str = ""
    relation_type: RelationType = RelationType.RELATED_TO
    confidence: float = 0.5
    source_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KGStats:
    """Statistics for the knowledge graph."""

    total_entities: int = 0
    total_relations: int = 0
    entity_type_counts: dict[str, int] = field(default_factory=dict)
    relation_type_counts: dict[str, int] = field(default_factory=dict)
    last_updated: datetime | None = None


# ── Entity extractor ────────────────────────────────────────────────────────


class EntityExtractor:
    """Extract entities from text using pattern matching.

    This is a rule-based extractor suitable for scientific text.
    A future version can delegate to an LLM for higher accuracy,
    but this must remain usable without external LLM calls (UMRS
    must work standalone).
    """

    # Common patterns for scientific entities
    _PATTERNS: list[tuple[str, EntityType]] = [
        # Software/tool names (CamelCase, known patterns)
        (r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", EntityType.SOFTWARE),
        # Method names with known suffixes
        (
            r"\b(\w+(?:Net|GAN|BERT|GPT|LLM|CNN|RNN|LSTM|VAE|ViT|SAM|MAE))\b",
            EntityType.METHOD,
        ),
        # Datasets (common naming patterns)
        (
            r"\b(ImageNet|COCO|CIFAR-?\d+|MNIST|WikiText|SQuAD|GLUE|SuperGLUE)\b",
            EntityType.DATASET,
        ),
        # Metrics
        (
            r"\b(accuracy|precision|recall|F1[-\s]?score|BLEU|ROUGE|perplexity|AUC|mAP)\b",
            EntityType.METRIC,
        ),
    ]

    def extract(self, text: str, doc_id: str = "") -> list[Entity]:
        """Extract entities from text."""
        entities: dict[str, Entity] = {}

        for pattern, etype in self._PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group(1)
                key = f"{etype.value}:{name.lower()}"
                if key in entities:
                    entities[key].mention_count += 1
                    if doc_id and doc_id not in entities[key].source_doc_ids:
                        entities[key].source_doc_ids.append(doc_id)
                else:
                    entities[key] = Entity(
                        name=name,
                        entity_type=etype,
                        source_doc_ids=[doc_id] if doc_id else [],
                    )

        return list(entities.values())


# ── Knowledge graph store (in-memory, can be backed by Postgres) ────────────


class KnowledgeGraph:
    """In-memory knowledge graph with entity and relation storage.

    Designed to be serializable to Postgres tables for persistence.
    The UMRS server can run a background worker that calls
    ``process_document()`` for each newly indexed document.
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._extractor = EntityExtractor()

    def process_document(self, doc_id: str, content: str) -> tuple[int, int]:
        """Extract entities and infer relations from a document.

        Returns:
            (entities_added, relations_added)
        """
        entities = self._extractor.extract(content, doc_id)
        new_entities = 0
        for entity in entities:
            if entity.entity_id in self._entities:
                existing = self._entities[entity.entity_id]
                existing.mention_count += entity.mention_count
                for did in entity.source_doc_ids:
                    if did not in existing.source_doc_ids:
                        existing.source_doc_ids.append(did)
            else:
                self._entities[entity.entity_id] = entity
                new_entities += 1

        # Infer co-occurrence relations
        new_relations = 0
        entity_ids = [e.entity_id for e in entities]
        for i, eid_a in enumerate(entity_ids):
            for eid_b in entity_ids[i + 1 :]:
                if eid_a != eid_b:
                    self._relations.append(
                        Relation(
                            source_entity_id=eid_a,
                            target_entity_id=eid_b,
                            relation_type=RelationType.RELATED_TO,
                            confidence=0.3,
                            source_doc_ids=[doc_id],
                        )
                    )
                    new_relations += 1

        return new_entities, new_relations

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def search_entities(
        self,
        query: str,
        entity_type: EntityType | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        """Search entities by name (case-insensitive substring)."""
        q = query.lower()
        results = []
        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            if q in entity.name.lower() or any(q in a.lower() for a in entity.aliases):
                results.append(entity)
        results.sort(key=lambda e: e.mention_count, reverse=True)
        return results[:limit]

    def get_related(
        self,
        entity_id: str,
        relation_type: RelationType | None = None,
        limit: int = 20,
    ) -> list[tuple[Entity, Relation]]:
        """Get entities related to the given entity."""
        results: list[tuple[Entity, Relation]] = []
        for rel in self._relations:
            if rel.source_entity_id == entity_id or rel.target_entity_id == entity_id:
                if relation_type and rel.relation_type != relation_type:
                    continue
                other_id = (
                    rel.target_entity_id
                    if rel.source_entity_id == entity_id
                    else rel.source_entity_id
                )
                other = self._entities.get(other_id)
                if other:
                    results.append((other, rel))
        results.sort(key=lambda x: x[1].confidence, reverse=True)
        return results[:limit]

    def stats(self) -> KGStats:
        """Return graph statistics."""
        entity_counts: dict[str, int] = {}
        for e in self._entities.values():
            key = e.entity_type.value
            entity_counts[key] = entity_counts.get(key, 0) + 1

        rel_counts: dict[str, int] = {}
        for r in self._relations:
            key = r.relation_type.value
            rel_counts[key] = rel_counts.get(key, 0) + 1

        return KGStats(
            total_entities=len(self._entities),
            total_relations=len(self._relations),
            entity_type_counts=entity_counts,
            relation_type_counts=rel_counts,
        )
