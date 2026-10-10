"""The dense null, calibrated in process (ADR-093 D-3, C2; ADR-083 D-5).

Thirty fixed probes, about nothing a project holds, give the mean and spread
of an unrelated query's best cosine in a scope. The values are cached by
(project, graph set, space, vector count), so a re-embed invalidates them,
and no table is written.
"""

from __future__ import annotations

import ast
import json
import statistics
import zlib
from pathlib import Path

import pytest
from umrs.knowledge.retrieval import calibrate
from umrs.knowledge.retrieval.calibrate import NULL_PROBES_V1, NullStats, null_for
from umrs.knowledge.retrieval.types import RetrieveSpec

_SCIFACT = Path(__file__).resolve().parents[2] / "benchmarks" / "retrieval" / "scifact.py"
_COSINES = [0.66 + 0.002 * i for i in range(30)]


class _Pool:
    """Answers the count and the probe statement; records every call."""

    def __init__(self, count=500, cosines=_COSINES):
        self.count, self.cosines, self.calls = count, cosines, []

    def getconn(self):
        return self

    def putconn(self, _conn):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        self.calls.append((sql, tuple(params or ())))
        if "unnest(" in sql:
            self.rows = [(i, 1.0 - c) for i, c in enumerate(self.cosines)]
        else:
            self.rows = [(self.count,)]

    def fetchall(self):
        return self.rows


class _Encoder:
    def __init__(self, model="/models/e5-base-v2", fail=False):
        self.model, self.fail, self.calls, self.batches = model, fail, 0, []

    def __call__(self, texts):
        self.calls += 1
        self.batches.append(list(texts))
        if self.fail:
            return "", None, "query encoder unavailable (ConnectError)"
        return self.model, [[1.0, float(i)] for i in range(len(texts))], ""

    @property
    def computed(self) -> int:
        """How many times the 30 probes were encoded (a null computed)."""
        return sum(1 for batch in self.batches if len(batch) == len(NULL_PROBES_V1))


@pytest.fixture(autouse=True)
def _empty_cache():
    calibrate.clear_cache()
    yield
    calibrate.clear_cache()


def _spec(**overrides):
    return RetrieveSpec(**{"queries": ("q",), "project_id": "p1", **overrides})


def _nonsense_eval_v1() -> tuple[str, ...]:
    tree = ast.parse(_SCIFACT.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "NONSENSE_EVAL_V1" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("NONSENSE_EVAL_V1 not found")


# ── the probe set ───────────────────────────────────────────────────────────


def test_thirty_distinct_probes():
    assert len(NULL_PROBES_V1) == 30 and len(set(NULL_PROBES_V1)) == 30
    assert calibrate.PROBES_VERSION == "null-v1"


@pytest.mark.skipif(not _SCIFACT.exists(), reason="benchmarks/retrieval not in this checkout")
def test_the_probes_are_disjoint_from_the_eval_nonsense_set():
    """Calibrating on the probes the gate scores would be circular."""
    evaluated = {p.strip().lower() for p in _nonsense_eval_v1()}
    assert evaluated and not evaluated & {p.strip().lower() for p in NULL_PROBES_V1}


# ── null_for ────────────────────────────────────────────────────────────────


def test_mu_and_sigma_are_the_probes_top_one_cosines():
    pool, encode = _Pool(), _Encoder()
    stats = null_for(pool, _spec(), encode, "/models/e5-base-v2", vectors=500)
    assert stats == NullStats(
        mu=pytest.approx(statistics.fmean(_COSINES)),
        sigma=pytest.approx(statistics.stdev(_COSINES)),
        n=30,
        model="e5-base-v2",
        probes_version="null-v1",
    )
    [(sql, params)] = pool.calls
    assert "unnest(%s::int[], %s::text[])" in sql and "n.content->>'kind' = 'chunk'" in sql
    assert params[-2] == list(range(30)) and len(params[-1]) == 30


def test_fewer_than_thirty_vectors_have_no_null():
    encode = _Encoder()
    assert null_for(_Pool(count=29), _spec(), encode, "m") is None
    assert encode.calls == 0


def test_the_null_is_cached_per_scope_and_a_re_embed_invalidates_it():
    pool, encode = _Pool(), _Encoder()
    first = null_for(pool, _spec(), encode, "/models/e5-base-v2", vectors=500)
    again = null_for(pool, _spec(), encode, "/opt/umrs/models/e5-base-v2", vectors=500)
    assert again is first and encode.computed == 1  # same space, same count
    null_for(pool, _spec(), encode, "/models/e5-base-v2", vectors=501)
    assert encode.computed == 2  # the vector count changed: recomputed
    null_for(pool, _spec(graph_ids=("g-1",)), encode, "/models/e5-base-v2", vectors=500)
    assert encode.computed == 3  # another graph set is another scope


def test_the_key_separates_projects_graph_sets_and_spaces():
    base = calibrate.scope_key(_spec(), "/models/e5-base-v2", 10)
    assert base[0] == "p1" and base[2:] == ("e5-base-v2", 10, None)
    assert calibrate.scope_key(_spec(), "/models/e5-base-v2", 10, "t2") != base
    assert (
        calibrate.scope_key(_spec(graph_ids=("b", "a")), "m", 10)[1]
        == calibrate.scope_key(_spec(graph_ids=("a", "b")), "m", 10)[1]
    )
    assert (
        calibrate.scope_key(_spec(graph_ids=()), "m", 10)[1]
        != calibrate.scope_key(_spec(), "m", 10)[1]
    )
    assert calibrate.scope_key(_spec(project_id="p2"), "/models/e5-base-v2", 10) != base


def test_an_encoder_outage_is_no_null_and_is_not_cached():
    pool = _Pool()
    assert null_for(pool, _spec(), _Encoder(fail=True), "/models/e5-base-v2", vectors=500) is None
    assert null_for(pool, _spec(), _Encoder(), "/models/e5-base-v2", vectors=500) is not None


def test_probes_from_another_space_are_refused():
    assert (
        null_for(
            _Pool(), _spec(), _Encoder(model="/models/bge-m3"), "/models/e5-base-v2", vectors=500
        )
        is None
    )


def test_the_count_is_read_when_the_caller_has_none():
    pool = _Pool(count=42)
    assert null_for(pool, _spec(), _Encoder(), "/models/e5-base-v2") is not None
    assert pool.calls[0][0].startswith("SELECT count(*)")


def test_the_lru_holds_at_most_its_size(monkeypatch):
    monkeypatch.setattr(calibrate, "CACHE_SIZE", 2)
    pool, encode = _Pool(), _Encoder()
    for count in (100, 101, 102):
        null_for(pool, _spec(), encode, "/models/e5-base-v2", vectors=count)
    assert len(calibrate._CACHE) == 2
    null_for(pool, _spec(), encode, "/models/e5-base-v2", vectors=100)
    assert encode.calls == 4  # the oldest was evicted


# ── the report CLI ──────────────────────────────────────────────────────────


def test_the_cli_prints_the_null_of_a_project(capsys):
    code = calibrate.main(["--project", "p1"], pool=_Pool(), encode=_Encoder())
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["project"] == "p1" and out["vectors"] == 500
    assert out["null"]["n"] == 30 and out["space"] == "e5-base-v2"


def test_the_cli_says_why_there_is_no_null(capsys):
    code = calibrate.main(["--project", "p1"], pool=_Pool(count=3), encode=_Encoder())
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["null"] is None and "fewer than 30" in out["reason"]


# ── review fix (2026-10-10): a same-count re-embed invalidates the null ─────


class _StampedPool(_Pool):
    """The count statement also returns the scope's last re-embed time."""

    def __init__(self, stamps):
        super().__init__()
        self.stamps = list(stamps)

    def execute(self, sql, params=None):
        super().execute(sql, params)
        if "unnest(" not in sql:
            self.rows = [(self.count, self.stamps.pop(0))]


def test_a_same_count_re_embed_recomputes_the_null():
    """core-11: re-embedding keeps the count; updated_at moves (store upsert)."""
    pool, encode = _StampedPool(["2026-10-10 01:00", "2026-10-10 02:00"]), _Encoder()
    null_for(pool, _spec(), encode, "/models/e5-base-v2")
    null_for(pool, _spec(), encode, "/models/e5-base-v2")
    assert encode.calls == 2
    count_sql = pool.calls[0][0]
    assert count_sql.startswith("SELECT count(*), max(e.updated_at) FROM")


def test_the_dense_lanes_stamp_keys_the_null_too():
    pool, encode = _Pool(), _Encoder()
    first = null_for(pool, _spec(), encode, "/models/e5-base-v2", vectors=500, stamp="t1")
    again = null_for(pool, _spec(), encode, "/models/e5-base-v2", vectors=500, stamp="t1")
    assert again is first and encode.computed == 1
    null_for(pool, _spec(), encode, "/models/e5-base-v2", vectors=500, stamp="t2")
    assert encode.computed == 2


# ── review fix (2026-10-10, fix-regressions-7): the query encoding ──────────
#
# UMRS_RAG_QUERY_PREFIX=fixed|b0 changes how the rag service encodes queries;
# `standalone.sh up` recreates only the rag container, so the umrs process
# kept a null computed under the other encoding (same project, graphs,
# space, count and stamp): 3 of 30 nonsense probes rose to partial.

E5 = "/opt/umrs/models/e5-base-v2"


class _TextEncoder:
    """Vectors from the text and the encoding (``shift``), like a prefix switch."""

    def __init__(self, shift=0.0, noise=0.0):
        self.shift, self.noise, self.batches = shift, noise, []

    def __call__(self, texts):
        self.batches.append(list(texts))
        return E5, [self.vector(t) for t in texts], ""

    def vector(self, text):
        return [1.0, self.shift + (zlib.crc32(text.encode()) % 100) / 100 + self.noise]

    @property
    def computed(self) -> int:
        return sum(1 for batch in self.batches if len(batch) == len(NULL_PROBES_V1))


def _other_cosines():
    return [0.70 + 0.001 * i for i in range(30)]


def test_a_changed_query_encoding_recomputes_the_null():
    fixed, b0 = _TextEncoder(shift=0.0), _TextEncoder(shift=0.5)
    first = null_for(_Pool(), _spec(), b0, E5, vectors=500, stamp="t1")
    again = null_for(_Pool(cosines=_other_cosines()), _spec(), fixed, E5, vectors=500, stamp="t1")
    assert fixed.computed == 1 and again != first
    assert again.mu == pytest.approx(statistics.fmean(_other_cosines()))
    # ... and the new encoding's null is the one cached now.
    assert null_for(_Pool(), _spec(), _TextEncoder(shift=0.0), E5, vectors=500, stamp="t1") is again


def test_the_same_encoding_keeps_the_cached_null_despite_float_noise():
    first = null_for(_Pool(), _spec(), _TextEncoder(), E5, vectors=500, stamp="t1")
    noisy = _TextEncoder(noise=1e-7)
    assert null_for(_Pool(), _spec(), noisy, E5, vectors=500, stamp="t1") is first
    assert noisy.computed == 0 and noisy.batches == [[calibrate.ENCODING_PROBE]]


def test_no_fingerprint_means_no_cached_null_is_trusted():
    null_for(_Pool(), _spec(), _Encoder(), "/models/e5-base-v2", vectors=500)
    down = _Encoder(fail=True)
    assert null_for(_Pool(), _spec(), down, "/models/e5-base-v2", vectors=500) is None


def test_a_fingerprinted_encoder_carries_the_probe_in_the_query_batch():
    inner = _TextEncoder()
    encode = calibrate.FingerprintedEncoder(inner)
    model, vectors, reason = encode(["q1", "q2"])
    assert inner.batches == [["q1", "q2", calibrate.ENCODING_PROBE]]
    assert (model, vectors, reason) == (E5, [inner.vector("q1"), inner.vector("q2")], "")
    assert encode.encoding_probe == inner.vector(calibrate.ENCODING_PROBE)
    first = null_for(_Pool(), _spec(), encode, E5, vectors=500, stamp="t1")
    assert inner.computed == 1 and inner.batches[-1] == list(NULL_PROBES_V1)  # never 31
    encode(["q3"])
    calls = len(inner.batches)
    assert null_for(_Pool(), _spec(), encode, E5, vectors=500, stamp="t1") is first
    assert len(inner.batches) == calls  # the hit needed no call of its own


def test_a_fingerprinted_encoder_passes_an_outage_or_a_short_answer_through():
    outage = calibrate.FingerprintedEncoder(_Encoder(fail=True))
    assert outage(["q"]) == ("", None, "query encoder unavailable (ConnectError)")
    assert outage.encoding_probe is None
    short = calibrate.FingerprintedEncoder(lambda _texts: (E5, [[1.0, 0.0]], ""))
    assert short(["q"]) == (E5, None, "query encoder returned 1 vectors for 2 texts")
