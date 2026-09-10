"""Cluster-venue loop entry — `python -m umrs.knowledge.cluster_loop_entry` (WP-05 round 2).

Runs one research loop's driver inside the cluster's mini-RiVault/Slurm
sandbox with per-job EgressGuard (upstream and UMRS gateway are almost the
only allowed destinations round 1). Loop truth stays server-side; the
driver only writes through the remote store adapter.

Currently drive mode is ``engine`` only (phased engine path); the agentic
driver's executor wiring is intentionally refused here with a named reason
(until the cluster-side companion executor is wired).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from . import loop_hosting
from .http_loop_store import build_cluster_store
from .loop import AutonomousResearchLoop

logger = logging.getLogger(__name__)


def _engine_factory_from(store: Any, *, loop: Any | None = None) -> Any:
    """Cluster-side engine factory.

    ``metadata.simulated`` on the loop record selects the sim engine (leg /
    probe loop records); anything else is the real phased engine.
    """
    metadata = getattr(loop, "metadata", {}) or {}
    simulated = bool(metadata.get("simulated"))

    def _make() -> AutonomousResearchLoop:
        if simulated:
            return AutonomousResearchLoop(simulated=True)
        return AutonomousResearchLoop(store)

    return _make


def _agent_denied_fabric(project_id: str) -> Any:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "cluster-venue agent driver not yet wired (engine only, round 2 "
            "honest limit; the companion executor needs a cluster-side "
            "client stack, which is follow-on)"
        )

    return _raise


def configure_cluster_hosting(loop: Any) -> None:
    """Wire the loop_hosting seam for the cluster driver."""
    store = build_cluster_store()
    simulate = bool((getattr(loop, "metadata", {}) or {}).get("simulated"))
    loop_hosting.configure(
        loop_hosting.HostingWiring(
            store=lambda: store,
            engine_factory=lambda: _engine_factory_from(store, loop=loop),
            agent_executor=_agent_denied_fabric,
            checkpoint_store=lambda: None,  # durable checkpoints stay server-side
            finding_sink=lambda: store,
        )
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-id", required=True)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--attempt-id", default="")
    args, extras = parser.parse_known_args(argv)

    try:
        store = build_cluster_store()
    except RuntimeError as exc:
        logger.error("cluster loop hosting cannot configure: %s", exc)
        return 2

    loop = store.load(args.loop_id)
    if loop.metadata.get("agent"):
        print(
            "cluster driver refused: agent driver not wired on cluster venues "
            "(engine only in round 2; loop recorded as engine anyway if "
            "the record says so)",
            file=sys.stderr,
        )
        # The loop's agent flag is honored by loop_hosting.host; refuse at
        # admission to avoid a misleading run.
        return 78

    configure_cluster_hosting(loop)
    kind = loop_hosting.host(args.loop_id, job_id=args.job_id, attempt_id=args.attempt_id)
    logger.info("cluster hosting finished: loop=%s kind=%s", args.loop_id, kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
