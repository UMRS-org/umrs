"""Research Sessions — durable investigation identity (ADR-025).

A session is an umbrella over one thread of research: loops started under it,
a title and goal, and an authoritative-runtime hint. UMRS owns the schema and
the REST surface; local (client-side, SQLite) loops are tracked by reference
only — venue + loop_id — never by state copy.

Follows the jobs/schema.py per-component pattern: ``RESEARCH_SESSION_SCHEMA_VERSION``
constant, ``ensure_schema(conn)`` tracked in the ``schema_versions`` table with
component ``research_sessions``.
"""

from .schema import RESEARCH_SESSION_SCHEMA_VERSION, ensure_schema
from .store import ResearchSession, SessionLoopLink, SessionStore

__all__ = [
    "RESEARCH_SESSION_SCHEMA_VERSION",
    "ResearchSession",
    "SessionLoopLink",
    "SessionStore",
    "ensure_schema",
]
