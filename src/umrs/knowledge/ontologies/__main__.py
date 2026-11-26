"""Print the bundled-ontology seed migration (ADR-081 KN-2, slice KN-S7).

Usage::

    python -m umrs.knowledge.ontologies > \\
        umrs/src/umrs/migrations/<next>_knowledge_terms_seed.sql

Offline: it reads only this package's YAML files.
"""

from __future__ import annotations

import sys

from ..term_closure import admitted_set, build_closure
from . import bundled_by_ref
from .seed_sql import render_seed_sql


def bundled_seed_sql() -> str:
    """The seed migration text for every bundled document."""
    docs = bundled_by_ref()
    terms = [term for doc in docs.values() for term in doc.terms]
    admitted = {ref: admitted_set(ref, docs) for ref in docs}
    return render_seed_sql(list(docs.values()), build_closure(terms), admitted)


def main() -> int:
    sys.stdout.write(bundled_seed_sql())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
