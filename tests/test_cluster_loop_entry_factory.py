"""Cluster-venue engine-factory pins (WP-05 r2 venue leg, 2026-09-17).

The first live cluster-venue submission failed at planning with
``AutonomousResearchLoop.__init__() takes 1 positional argument but 2 were
given`` — the WP-14-era factory passed the loop store positionally into a
constructor that has been keyword-only since before WP-14, and no test ever
invoked the factory body. These pins lock the constructor-callable contract
for both branches so the class can't regress silently again.
"""
from types import SimpleNamespace

from umrs.knowledge.cluster_loop_entry import _engine_factory_from
from umrs.knowledge.loop import AutonomousResearchLoop


def _loop_meta(simulated):
    return SimpleNamespace(metadata={"simulated": simulated})


def test_real_branch_builds_a_real_engine():
    make = _engine_factory_from(object(), loop=_loop_meta(False))
    engine = make()
    assert isinstance(engine, AutonomousResearchLoop)
    # contract: the phased engine (no simulate flag on the build)
    assert getattr(engine, "_simulated") is False


def test_sim_branch_builds_simulated_engine():
    make = _engine_factory_from(object(), loop=_loop_meta(True))
    engine = make()
    assert isinstance(engine, AutonomousResearchLoop)
    assert getattr(engine, "_simulated") is True


def test_missing_metadata_defaults_to_real():
    make = _engine_factory_from(object(), loop=SimpleNamespace(metadata={}))
    assert getattr(make(), "_simulated") is False
