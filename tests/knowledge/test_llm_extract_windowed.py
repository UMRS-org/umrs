"""ADR-116 D-2/D-6: whole-document windowed extraction (knowledge/llm_extract).

The model is a fake chat callable; no network, no model.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.knowledge import llm_extract as lx


class FakeVocab:
    """The [C-ONT] ExtractionVocabulary surface, two classes and one relation."""

    ontology_id = "test-onto"
    version = "1"

    def class_names(self) -> list[str]:
        return ["material", "method"]

    def relation_names(self) -> list[str]:
        return ["uses"]

    def prompt_block(self) -> str:
        return "Classes: material, method.\nRelations: uses (method -> material)."

    def match_class(self, raw: str) -> str | None:
        name = raw.strip().lower().removesuffix("s")
        return name if name in ("material", "method") else None

    def match_relation(self, raw: str) -> str | None:
        return "uses" if raw.strip().lower() in ("uses", "use", "applies") else None

    def bucket_for(self, cls: str) -> str:
        return "entity" if cls == "material" else "concept"

    def gliner_labels(self) -> list[str]:
        return ["material", "method"]


def _sentences(count: int, word: str = "Filler") -> str:
    return " ".join(f"{word} sentence number {i} says little." for i in range(count))


def _settings(**kw: Any) -> lx.ExtractSettings:
    base = {"window_chars": 2000, "windows_max": 24, "concurrency": 4, "budget_s": 60.0}
    return lx.ExtractSettings(**{**base, **kw})


class Chat:
    """Answers each window with the entities whose markers it contains."""

    def __init__(self, replies: dict[str, dict[str, Any]] | None = None) -> None:
        self.replies = replies or {}
        self.calls: list[dict[str, Any]] = []

    def __call__(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        user = messages[-1]["content"]
        self.calls.append({"user": user, "system": messages[0]["content"], **kw})
        entities: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        for marker, reply in self.replies.items():
            if marker in user:
                entities += reply.get("entities", [])
                relations += reply.get("relations", [])
        return json.dumps({"entities": entities, "relations": relations})


def _request(chat: Any, **kw: Any) -> lx.ExtractRequest:
    return lx.ExtractRequest(
        chat=chat,
        model="m",
        vocab=kw.pop("vocab", FakeVocab()),
        settings=kw.pop("settings", _settings()),
        **kw,
    )


# ── windows ─────────────────────────────────────────────────────────────────


def test_windows_cover_the_whole_text_within_the_size():
    text = _sentences(400)
    windows = lx.pack_windows(text, 2000)
    assert len(windows) > 5
    assert max(len(w.text) for w in windows) <= 2000
    assert [w.index for w in windows] == list(range(len(windows)))
    assert windows[-1].text.endswith("399 says little.")
    # the windows are slices of the text, and only whitespace lies between them
    rebuilt = " ".join(w.text for w in windows)
    assert rebuilt.split() == text.split()


def test_a_sentence_longer_than_a_window_is_cut():
    text = "x" * 5000 + "."
    windows = lx.pack_windows(text, 2000)
    assert [len(w.text) for w in windows] == [2000, 2000, 1001]


def test_empty_text_has_no_window():
    assert lx.pack_windows("   \n ", 2000) == []


def test_settings_clamp_concurrency_and_read_the_env(monkeypatch):
    monkeypatch.setenv(lx.CONCURRENCY_ENV, "50")
    monkeypatch.setenv(lx.WINDOW_CHARS_ENV, "nonsense")
    monkeypatch.setenv(lx.BUDGET_ENV, "12")
    settings = lx.ExtractSettings.from_env()
    assert settings.concurrency == 10
    assert settings.window_chars == 8000
    assert settings.budget_s == 12.0
    monkeypatch.setenv(lx.CONCURRENCY_ENV, "0")
    assert lx.ExtractSettings.from_env().concurrency == 1


def test_defaults():
    settings = lx.ExtractSettings()
    assert (settings.window_chars, settings.windows_max, settings.concurrency) == (8000, 24, 4)
    assert (settings.budget_s, settings.job_budget_s, settings.max_tokens) == (300.0, 300.0, 2500)


def test_the_job_budget_reads_its_env(monkeypatch):
    monkeypatch.setenv(lx.JOB_BUDGET_ENV, "42")
    assert lx.ExtractSettings.from_env().job_budget_s == 42.0


# ── reply parsing ───────────────────────────────────────────────────────────

_OBJ = {"entities": [{"name": "Zirconia", "type": "material"}], "relations": []}


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps(_OBJ),
        "```json\n" + json.dumps(_OBJ) + "\n```",
        "<think>maybe {not json} here</think>\n" + json.dumps(_OBJ),
        "reasoning without an opening tag</think>" + json.dumps(_OBJ),
        "Here you go: " + json.dumps(_OBJ) + " trailing {text}",
        "{broken " + json.dumps(_OBJ),
    ],
)
def test_reply_parser_finds_the_object(reply):
    assert lx.parse_reply(reply) == _OBJ


def test_reply_parser_keeps_braces_inside_strings():
    obj = {"entities": [{"name": "set {a, b}", "type": "x"}]}
    assert lx.parse_reply("prefix " + json.dumps(obj)) == obj


@pytest.mark.parametrize("reply", [None, "", "   ", "no json at all", "[1, 2]", "{", 42])
def test_reply_parser_never_raises_on_garbage(reply):
    assert lx.parse_reply(reply) is None


# ── merge and mapping ───────────────────────────────────────────────────────


def _result(index: int, data: dict[str, Any]) -> lx.WindowResult:
    return lx.WindowResult(lx.Window(index, 0, 0, ""), data)


def test_merge_by_normalized_name_unions_aliases_and_evidence():
    results = [
        _result(
            0,
            {
                "entities": [
                    {"name": "Zirconia", "type": "material", "quote": "q0", "aliases": ["ZrO2"]}
                ]
            },
        ),
        _result(1, {"entities": [{"name": "zirconia", "type": "material", "quote": "q1"}]}),
        _result(2, {"entities": [{"name": "ZrO2", "type": "material", "quote": "q2"}]}),
    ]
    payload = lx.merge_windows(results, FakeVocab(), "pypdf")
    assert payload["extractor"] == "llm-windowed@2"
    assert payload["nodes"] == [
        {
            "name": "Zirconia",
            "type": "material",
            "bucket": "entity",
            "aliases": ["ZrO2", "zirconia"],
            "evidence": [
                {"window": 0, "quote": "q0", "parser": "pypdf"},
                {"window": 1, "quote": "q1", "parser": "pypdf"},
                {"window": 2, "quote": "q2", "parser": "pypdf"},
            ],
        }
    ]


def test_evidence_is_capped_at_five_and_quotes_at_200_chars():
    results = [
        _result(i, {"entities": [{"name": "A", "type": "method", "quote": f"{i}" + "x" * 500}]})
        for i in range(8)
    ]
    (node,) = lx.merge_windows(results, FakeVocab(), "raw")["nodes"]
    assert len(node["evidence"]) == 5
    assert all(len(e["quote"]) <= 200 for e in node["evidence"])


def test_first_type_wins_unless_a_later_one_matches_the_vocabulary():
    results = [
        _result(0, {"entities": [{"name": "Sintering", "type": "procedure"}]}),
        _result(1, {"entities": [{"name": "Sintering", "type": "method"}]}),
        _result(2, {"entities": [{"name": "Sintering", "type": "material"}]}),
    ]
    (node,) = lx.merge_windows(results, FakeVocab(), "raw")["nodes"]
    assert node["type"] == "method"


def test_unmatched_type_is_unclassified_with_type_raw():
    results = [_result(0, {"entities": [{"name": "Dr. X", "type": "Person"}]})]
    (node,) = lx.merge_windows(results, FakeVocab(), "raw")["nodes"]
    assert node["type"] == "unclassified"
    assert node["type_raw"] == "Person"
    assert node["bucket"] == "concept"


def test_relations_map_to_the_vocabulary_or_related_to():
    results = [
        _result(
            0,
            {
                "entities": [
                    {"name": "Sintering", "type": "method"},
                    {"name": "Zirconia", "type": "material"},
                ],
                "relations": [
                    {"source": "Sintering", "target": "zirconia", "relation": "Uses"},
                    {"source": "Zirconia", "target": "Sintering", "relation": "resists"},
                    {"source": "Zirconia", "target": "ghost", "relation": "uses"},
                    {"source": "Zirconia", "target": "Zirconia", "relation": "uses"},
                ],
            },
        )
    ]
    edges = lx.merge_windows(results, FakeVocab(), "raw")["edges"]
    assert {"source": "Sintering", "target": "Zirconia", "relation": "uses"} in edges
    assert {
        "source": "Zirconia",
        "target": "Sintering",
        "relation": "related_to",
        "rel_raw": "resists",
    } in edges
    assert len(edges) == 2


def test_a_relation_across_windows_resolves_through_aliases():
    results = [
        _result(0, {"entities": [{"name": "Zirconia", "type": "material", "aliases": ["ZrO2"]}]}),
        _result(
            1,
            {
                "entities": [{"name": "Sintering", "type": "method"}],
                "relations": [{"source": "Sintering", "target": "ZrO2", "relation": "uses"}],
            },
        ),
    ]
    edges = lx.merge_windows(results, FakeVocab(), "raw")["edges"]
    assert edges == [{"source": "Sintering", "target": "Zirconia", "relation": "uses"}]


def test_builtin_vocabulary_matches_todays_types():
    vocab = lx.BuiltinVocabulary()
    assert vocab.match_class("Datasets") == "dataset"
    assert vocab.match_class("gizmo") is None
    assert vocab.bucket_for("software") == "entity"
    assert vocab.bucket_for("method") == "concept"
    assert vocab.match_relation("Part of") == "part_of"


# ── the driver ──────────────────────────────────────────────────────────────


def test_the_prompt_lists_the_vocabulary_and_asks_for_json_mode():
    chat = Chat({"Filler": {"entities": [{"name": "Zirconia", "type": "material"}]}})
    notes: list[str] = []
    payload = lx.extract_document(_sentences(5), _request(chat), notes)
    assert payload is not None and payload["nodes"][0]["name"] == "Zirconia"
    assert "Classes: material, method." in chat.calls[0]["system"]
    assert chat.calls[0]["response_format"] == {"type": "json_object"}
    assert notes == []


def test_every_window_is_sent_and_merged():
    text = _sentences(60) + " The ALPHA phase appears. " + _sentences(60, "Tail") + " OMEGA."
    chat = Chat(
        {
            "ALPHA": {"entities": [{"name": "Alpha phase", "type": "material", "quote": "ALPHA"}]},
            "OMEGA": {"entities": [{"name": "Omega", "type": "method"}]},
        }
    )
    payload = lx.extract_document(text, _request(chat), [])
    assert len(chat.calls) == len(lx.pack_windows(text, 2000)) > 2
    assert {n["name"] for n in payload["nodes"]} == {"Alpha phase", "Omega"}


class _HttpError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


def test_json_mode_refused_with_4xx_is_retried_once_without_it():
    seen: list[Any] = []

    def chat(messages: list[dict[str, Any]], **kw: Any) -> str:
        seen.append(kw.get("response_format"))
        if kw.get("response_format") is not None:
            raise _HttpError(400)
        return json.dumps(_OBJ)

    text = _sentences(200)
    request = _request(chat, settings=_settings(concurrency=1))
    payload = lx.extract_document(text, request, [])
    assert payload is not None
    windows = len(lx.pack_windows(text, 2000))
    # one refused JSON-mode request, then plain requests only
    assert seen[:2] == [{"type": "json_object"}, None]
    assert seen.count({"type": "json_object"}) == 1
    assert len(seen) == windows + 1


def test_a_5xx_is_not_retried_and_is_named():
    def chat(messages: list[dict[str, Any]], **kw: Any) -> str:
        raise _HttpError(503)

    notes: list[str] = []
    assert lx.extract_document(_sentences(3), _request(chat), notes) is None
    assert notes and notes[0].startswith("semantic extraction unavailable: _HttpError")


def test_reasoning_only_replies_fail_by_name_not_by_exception():
    notes: list[str] = []
    assert lx.extract_document(_sentences(3), _request(lambda *_a, **_k: ""), notes) is None
    assert "replied without a JSON graph" in notes[0]


def test_cap_names_the_skipped_windows():
    text = _sentences(200)
    chat = Chat({"Filler": {"entities": [{"name": "A", "type": "method"}]}})
    notes: list[str] = []
    lx.extract_document(text, _request(chat, settings=_settings(windows_max=2)), notes)
    total = len(lx.pack_windows(text, 2000))
    assert len(chat.calls) == 2
    assert f"extract_windows_skipped:{total - 2}" in notes


def test_lite_reads_one_window_and_names_nothing():
    text = _sentences(200)
    chat = Chat({"Filler": {"entities": [{"name": "A", "type": "method"}]}})
    notes: list[str] = []
    payload = lx.extract_document(text, _request(chat, mode="lite"), notes)
    assert payload is not None
    assert len(chat.calls) == 1
    assert notes == []


def test_budget_exhaustion_is_named():
    def slow(messages: list[dict[str, Any]], **kw: Any) -> str:
        if "number 0 " not in messages[-1]["content"]:
            time.sleep(1.5)
        return json.dumps(_OBJ)

    notes: list[str] = []
    request = _request(slow, settings=_settings(budget_s=1.0, concurrency=1))
    payload = lx.extract_document(_sentences(200), request, notes)
    assert payload is not None  # the first window finished in time
    assert "extract_budget_exhausted" in notes
    assert any(n.startswith("extract_windows_skipped:") for n in notes)


def test_tagger_candidates_go_into_the_prompt_as_hints():
    chat = Chat()

    def tagger(texts: list[str]) -> list[list[dict[str, Any]]]:
        return [[{"text": "Zirconia", "label": "material", "score": 0.9}] for _ in texts]

    lx.extract_document(_sentences(3), _request(chat, tagger=tagger), [])
    assert "- Zirconia (material)" in chat.calls[0]["user"]


def test_a_failing_tagger_does_not_stop_extraction():
    def tagger(texts: list[str]) -> list[list[dict[str, Any]]]:
        raise RuntimeError("boom")

    chat = Chat({"Filler": {"entities": [{"name": "A", "type": "method"}]}})
    assert lx.extract_document(_sentences(3), _request(chat, tagger=tagger), []) is not None


def test_run_bounded_keeps_job_order():
    results, late = lx.run_bounded([lambda i=i: i * i for i in range(7)], 3, 5.0)
    assert results == [0, 1, 4, 9, 16, 25, 36] and late == 0


# ── session-history packages ────────────────────────────────────────────────


def test_session_history_is_read_as_turn_texts():
    package = json.dumps(
        {
            "lane": "session-history",
            "built_by": "agent",
            "lines": [
                {"role": "user", "text": "Which zirconia phase?", "verbatim": True},
                {"role": "assistant", "text": "Tetragonal.", "verbatim": True},
                {"role": "assistant", "text": "  "},
            ],
        }
    )
    assert lx.session_history_text(package) == (
        "user: Which zirconia phase?\nassistant: Tetragonal."
    )


@pytest.mark.parametrize(
    "text", ["plain prose", json.dumps({"lane": "other", "lines": []}), "{not json"]
)
def test_other_text_is_not_session_history(text):
    assert lx.session_history_text(text) is None


# ── vocabulary resolution ───────────────────────────────────────────────────


def test_vocabulary_resolution_uses_the_ontology_module(monkeypatch):
    import sys

    vocab = FakeVocab()
    fake = SimpleNamespace(
        resolve_graph_vocabulary=lambda p, g: (vocab, [f"ontology_fallback:{p}:{g}"]),
        default_vocabulary=lambda: vocab,
    )
    monkeypatch.setitem(sys.modules, "umrs.knowledge.ontology_vocab", fake)
    assert lx.resolve_vocabulary("p", "g") == (vocab, ["ontology_fallback:p:g"])
    assert lx.default_vocabulary() is vocab


def test_vocabulary_resolution_never_raises(monkeypatch):
    import sys

    def boom(_p: str, _g: str) -> Any:
        raise RuntimeError("db down")

    fake = SimpleNamespace(resolve_graph_vocabulary=boom, default_vocabulary=FakeVocab)
    monkeypatch.setitem(sys.modules, "umrs.knowledge.ontology_vocab", fake)
    vocab, notes = lx.resolve_vocabulary("p", "g")
    assert isinstance(vocab, FakeVocab)
    assert notes == ["ontology_fallback:RuntimeError"]


def test_vocabulary_module_missing_falls_back_to_builtin(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "umrs.knowledge.ontology_vocab", None)
    vocab, notes = lx.resolve_vocabulary("p", "")
    assert isinstance(vocab, lx.BuiltinVocabulary)
    assert notes == ["ontology_fallback:vocabulary_module_missing"]


# ── replies without a graph (review fix: no silent empty result) ────────────

_TRUNCATED = '{"entities": [{"name": "graphene", "type": "material", "quote": "x"}, {"name": "copp'


@pytest.mark.parametrize(
    "reply",
    [_TRUNCATED, '{"name": "graphene", "type": "material"}', '{"answer": "none"}'],
)
def test_a_reply_without_a_graph_list_is_not_a_graph(reply):
    assert lx.parse_reply(reply) is None


def test_a_wrapped_graph_is_found_inside_its_wrapper():
    assert lx.parse_reply(json.dumps({"result": _OBJ})) == _OBJ


def test_a_truncated_reply_fails_the_window_by_name():
    notes: list[str] = []
    request = _request(lambda *_a, **_k: _TRUNCATED)
    assert lx.extract_document(_sentences(600), request, notes) is None
    assert notes[0].startswith("semantic extraction unavailable:")
    assert "without a JSON graph" in notes[0]


def test_some_truncated_windows_are_counted_as_failed():
    calls: list[int] = []

    def chat(messages: list[dict[str, Any]], **_kw: Any) -> str:
        calls.append(1)
        first = "part 1 of" in messages[-1]["content"]
        return json.dumps(_OBJ) if first else _TRUNCATED

    notes: list[str] = []
    payload = lx.extract_document(_sentences(200), _request(chat), notes)
    assert payload is not None
    assert [n["name"] for n in payload["nodes"]] == ["Zirconia"]
    assert f"extract_windows_failed:{len(calls) - 1}" in notes


def test_label_or_id_stands_in_for_a_missing_name():
    data = {"entities": [{"label": "Ceria", "type": "material"}, {"id": "XRD"}]}
    payload = lx.merge_windows([_result(0, data)], FakeVocab(), "raw")
    assert [n["name"] for n in payload["nodes"]] == ["Ceria", "XRD"]


def test_a_single_key_map_entity_reads_as_name_and_type():
    """qwen2.5:14b answered ``{"entities": [{"GLiNER": "Model"}]}`` live."""
    data = {"entities": [{"GLiNER": "Model"}, {"Ceria": 3}, {"name": "XRD"}]}
    payload = lx.merge_windows([_result(0, data)], FakeVocab(), "raw")
    nodes = {n["name"]: n.get("type_raw") for n in payload["nodes"]}
    assert nodes == {"GLiNER": "Model", "Ceria": "", "XRD": ""}


@pytest.mark.parametrize("bad", ["", "no json here", _TRUNCATED])
def test_unparsed_windows_are_counted_by_name(bad):
    def chat(messages: list[dict[str, Any]], **_kw: Any) -> str:
        return json.dumps(_OBJ) if "part 1 of" in messages[-1]["content"] else bad

    notes: list[str] = []
    payload = lx.extract_document(_sentences(200), _request(chat), notes)
    total = len(lx.pack_windows(_sentences(200), 2000))
    assert payload is not None
    assert f"extract_window_unparsed:{total - 1}" in notes


def test_a_failed_call_is_not_counted_as_unparsed():
    def chat(messages: list[dict[str, Any]], **_kw: Any) -> str:
        if "part 1 of" not in messages[-1]["content"]:
            raise TimeoutError("read timeout")
        return json.dumps(_OBJ)

    notes: list[str] = []
    assert lx.extract_document(_sentences(200), _request(chat), notes) is not None
    assert not any(n.startswith("extract_window_unparsed:") for n in notes)


def test_a_document_past_the_job_budget_reads_one_window():
    chat = Chat({"Filler": {"entities": [{"name": "A", "type": "method"}]}})
    notes: list[str] = []
    request = _request(chat, deadline=time.monotonic() - 1.0)
    payload = lx.extract_document(_sentences(200), request, notes)
    assert payload is not None and len(chat.calls) == 1
    assert notes == ["extract_job_budget_exhausted"]


def test_the_job_deadline_cuts_the_document_budget(monkeypatch):
    def slow(messages: list[dict[str, Any]], **_kw: Any) -> str:
        if "number 0 " not in messages[-1]["content"]:
            time.sleep(1.5)
        return json.dumps(_OBJ)

    monkeypatch.setattr(lx, "LITE_FLOOR_S", 0.1)
    notes: list[str] = []
    request = _request(
        slow, settings=_settings(budget_s=60.0, concurrency=1), deadline=time.monotonic() + 0.8
    )
    started = time.monotonic()
    assert lx.extract_document(_sentences(200), request, notes) is not None
    assert time.monotonic() - started < 1.4
    assert "extract_budget_exhausted" in notes
