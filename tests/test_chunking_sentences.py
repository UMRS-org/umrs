"""ADR-093 D-11 (C9): the ``sent-v1`` chunk scheme.

A deterministic regex splitter (abbreviation guard, headings, blank lines,
list items, CJK end marks), packing to 200 tokens (max 256, one chunk up to
384, one-sentence overlap, tail merge-back, oversize windows), offsets into
the original text, a ``{title} › {section}`` header with no LLM — and the
two seams wave A relies on: hydrate's ``search_text`` md5 rule and the
worker's ``passage: `` + header + newline + body embedding input.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from itertools import pairwise
from types import SimpleNamespace

import pytest
from umrs.knowledge import chunking as ch
from umrs.knowledge import embed_worker as ew
from umrs.knowledge.retrieval.hydrate import stored_text

E5 = "/opt/umrs/models/e5-base-v2"


def _sentences(text: str) -> list[str]:
    return [text[a:b] for a, b in ch.sentence_spans(text)]


def _prose(count: int, words: int = 7, first: int = 0) -> str:
    """``count`` sentences of ``words`` words and a full stop (``words + 1``
    approximate tokens each)."""
    return " ".join(
        " ".join(f"w{first + i}x{j}" for j in range(words)).capitalize() + "." for i in range(count)
    )


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── the splitter ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The value was 3.14 at pH 7.4. It rose.", ["The value was 3.14 at pH 7.4.", "It rose."]),
        (
            "See Fig. 2 and Eq. (3) for details. Then stop.",
            ["See Fig. 2 and Eq. (3) for details.", "Then stop."],
        ),
        (
            "Smith et al. Found more. Dr. Jones agreed.",
            ["Smith et al. Found more.", "Dr. Jones agreed."],
        ),
        ("J. Smith wrote it. E.g. this one.", ["J. Smith wrote it.", "E.g. this one."]),
        (
            "Item No. 5 was best. He said no. Next one.",
            ["Item No. 5 was best.", "He said no.", "Next one."],
        ),
        ("Is it? Yes! Done.", ["Is it?", "Yes!", "Done."]),
        (
            "Values (e.g. 1.5 mM) were used. U.S. data agree.",
            ["Values (e.g. 1.5 mM) were used.", "U.S. data agree."],
        ),
        ('He said "stop." Then left.', ['He said "stop."', "Then left."]),
    ],
)
def test_sentence_ends_with_an_abbreviation_guard(text, expected):
    """Decimals, Fig./Eq./et al./Dr., initials and initialisms, and "No."
    before a number never end a sentence; a closing quote stays with it."""
    assert _sentences(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "これは文です。次の文；最後の文！終わり？",
            ["これは文です。", "次の文；", "最後の文！", "終わり？"],
        ),
        ("結果は良い。The end.", ["結果は良い。", "The end."]),
    ],
)
def test_cjk_end_marks_split(text, expected):
    assert _sentences(text) == expected


def test_headings_blank_lines_and_list_items_are_boundaries():
    text = (
        "# Title\nIntro line\n\nPara two\n- item one\n  continues here\n"
        "- item two\n1. first\n2) second"
    )
    assert _sentences(text) == [
        "# Title",
        "Intro line",
        "Para two",
        "- item one\n  continues here",
        "- item two",
        "1. first",
        "2) second",
    ]


# ── packing ─────────────────────────────────────────────────────────────────


def test_a_body_of_384_tokens_stays_one_chunk():
    text = _prose(48)  # 48 x 8 = 384 tokens
    chunks = ch.sentence_chunks(text)
    assert [(c["start"], c["end"], c["num_tokens"]) for c in chunks] == [(0, len(text), 384)]
    assert len(ch.sentence_chunks(_prose(49))) == 2  # 392 tokens: packed


def test_packing_targets_200_tokens_and_overlaps_one_short_sentence():
    text = _prose(150)  # 1,200 tokens
    chunks = ch.sentence_chunks(text)
    assert [c["num_tokens"] for c in chunks[:-1]] == [200] * (len(chunks) - 1)
    for prev, nxt in pairwise(chunks):
        overlap = text[nxt["start"] : prev["end"]]
        assert _sentences(overlap) == [_sentences(prev["text"])[-1]]


def test_a_sentence_over_48_tokens_is_not_repeated():
    text = _prose(20, words=59)  # 60-token sentences
    chunks = ch.sentence_chunks(text)
    assert [c["num_tokens"] for c in chunks] == [240] * 5
    assert all(nxt["start"] > prev["end"] for prev, nxt in pairwise(chunks))


def test_a_tail_under_48_tokens_merges_back():
    """52 sentences: a third chunk would own 24 tokens; it merges into the
    second instead."""
    text = _prose(52)
    chunks = ch.sentence_chunks(text)
    assert [c["num_tokens"] for c in chunks] == [200, 224]
    assert chunks[-1]["end"] == len(text)


def _sentence(tokens: int) -> str:
    """One sentence of ``tokens`` approximate tokens (words and a stop)."""
    return " ".join(f"v{j}" for j in range(tokens - 1)).capitalize() + "."


@pytest.mark.parametrize(
    "sizes",
    [
        (40, 250, 250, 40),  # a short opening sentence carried a long one to 290
        (220, 240, 30),  # the tail merged back to 270
    ],
)
def test_a_packed_chunk_never_passes_the_maximum(sizes):
    """Review chunk-data-7: C9's packing maximum is 256 tokens; only a body
    of 384 tokens or less (one chunk) is larger."""
    chunks = ch.sentence_chunks(" ".join(_sentence(n) for n in sizes))
    assert len(chunks) > 1
    assert max(c["num_tokens"] for c in chunks) <= ch.MAX_TOKENS


def _random_text(rng: random.Random) -> str:
    sizes = (3, 30, 47, 48, 60, 120, 200, 250, 256, 300)
    return " ".join(_sentence(rng.choice(sizes)) for _ in range(rng.randint(2, 12)))


def test_no_packed_chunk_of_a_seeded_corpus_passes_the_maximum():
    rng = random.Random(93)
    texts = [t for t in (_random_text(rng) for _ in range(400)) if len(ch.TokenIndex(t)) > 384]
    largest = [max(c["num_tokens"] for c in ch.sentence_chunks(t)) for t in texts]
    assert len(largest) > 100 and max(largest) <= ch.MAX_TOKENS


def test_an_oversize_sentence_falls_back_to_token_windows():
    text = ("word " * 700).strip()  # one 700-token "sentence"
    chunks = ch.sentence_chunks(text)
    assert [c["num_tokens"] for c in chunks] == [256, 256, 188]
    assert all(nxt["start"] > prev["end"] for prev, nxt in pairwise(chunks))


MESSY = (
    "  # Report\r\n\r\nThe yield was 2.5 g. Fig. 3 shows it.\r\n\r\n"
    "## 結果\n実験は成功した。次に進む；\n\n- a point\n- b point\n\n"
    + _prose(60, first=100)
    + "\n\n## Tail\n"
    + ("x " * 300)
)


def test_chunk_text_is_its_slice_of_the_original_text():
    chunks = ch.sentence_chunks(MESSY, title="Report")
    assert len(chunks) > 2
    assert all(c["text"] == MESSY[c["start"] : c["end"]] for c in chunks)
    assert all(c["md5"] == _md5(c["text"]) for c in chunks)


def test_chunking_is_deterministic():
    assert ch.sentence_chunks(MESSY, title="R") == ch.sentence_chunks(MESSY, title="R")


class _WordTokenizer:
    """A model tokenizer stand-in: one token per whitespace-separated word."""

    def __call__(self, text, **_options):
        offsets, cursor = [], 0
        for word in text.split():
            start = text.index(word, cursor)
            cursor = start + len(word)
            offsets.append((start, cursor))
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def test_token_counts_come_from_the_model_tokenizer_when_given():
    text = _prose(60)  # 420 words, 480 approximate tokens
    chunks = ch.sentence_chunks(text, tokenizer=_WordTokenizer())
    assert sum(len(c["text"].split()) for c in chunks) > 420  # overlap repeats words
    assert [c["num_tokens"] for c in chunks] == [len(c["text"].split()) for c in chunks]


# ── headers ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("title", "section", "header"),
    [
        ("Paper", "Methods", "Paper › Methods"),
        ("Paper", "paper", "Paper"),  # the same text once
        ("Paper", "", "Paper"),
        ("", "Methods", "Methods"),
        ("  Two\n lines ", "", "Two lines"),
    ],
)
def test_header_is_title_and_section(title, section, header):
    assert ch.build_header(title, section) == header


@pytest.mark.parametrize("section", ["Methods", " ".join(f"sec{i}" for i in range(40))])
def test_header_stays_within_32_tokens(section):
    title = " ".join(f"word{i}" for i in range(60))
    header = ch.build_header(title, section)
    assert len(ch.TokenIndex(header)) <= ch.HEADER_TOKENS
    assert ch.HEADER_SEP in header and header.startswith("word0 ")


@pytest.mark.parametrize(
    ("meta", "text", "name", "title"),
    [
        ("Meta title", "# Heading\nbody", "a.md", "Meta title"),
        ("", "\n  # Paper title #\n\nbody", "a.md", "Paper title"),
        ("", "# Untitled note\nbody", "", "Untitled note"),
        ("", "no heading at all", "a.md", "a.md"),
    ],
)
def test_title_is_metadata_then_leading_h1_then_source_name(meta, text, name, title):
    assert ch.pick_title(meta, text, name) == title


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Intro text. More.\n\n## Methods\nX.", "notes.md"),  # a later section
        ("intro\n\n## Heading two ##\nbody", "a.md"),
        ("## Sub only\nbody", "a.md"),  # an opening H2 names a section
        ("# Copyright 2025\nimport os\n", "plot.py"),  # a comment in code
        ("# Copyright (c) 2025 RIKEN\n# SPDX-License-Identifier: MIT\n", "run.sh"),
        ("# settings\nkey: value\n", "config.yaml"),
        ("# not a heading in plain text\nline\n", "notes.txt"),
    ],
)
def test_a_comment_or_a_later_heading_never_titles_the_chunks(text, name):
    """Review chunk-data-8: the first '#' line anywhere was the title, so a
    licence comment labelled every chunk of a script and a later '## Methods'
    labelled a note's introduction; the file name tells chunks apart."""
    assert ch.pick_title("", text, name) == name


@pytest.mark.parametrize(("name", "headings"), [("a.md", True), ("", True), ("b.py", False)])
def test_heading_syntax_follows_the_file_type(name, headings):
    assert ch.has_heading_syntax(name) is headings


def test_without_heading_syntax_a_comment_line_opens_no_section():
    code = "# Copyright 2025\nimport os\n\n# helpers\nx = 1\n"
    as_code = ch.sentence_chunks(code, title="plot.py", headings=False)
    as_markdown = ch.sentence_chunks(code, title="plot.py")
    assert [c["section"] for c in as_code] == [""]
    assert as_code[0]["header"] == "plot.py"
    assert as_markdown[0]["section"] == "Copyright 2025"  # markdown: a heading


def test_sections_follow_headings_and_a_heading_starts_a_chunk():
    text = f"# Paper\n\n{_prose(40)}\n\n## Methods\n\n{_prose(40, first=50)}"
    chunks = ch.sentence_chunks(text, title="Paper")
    methods = [c for c in chunks if c["text"].startswith("## Methods")]
    assert (chunks[0]["section"], chunks[0]["header"]) == ("", "Paper")
    assert len(methods) == 1
    assert (methods[0]["section"], methods[0]["header"]) == ("Methods", "Paper › Methods")


# ── the seams: hydrate's md5 rule and the worker's embedding input ──────────


def _hydrated(chunk: dict) -> str | None:
    search_text, header_chars = ch.header_search_text(chunk["header"], chunk["text"])
    content = {
        "chunk_start": chunk["start"],
        "chunk_end": chunk["end"],
        "chunk_md5": _md5(chunk["text"]),
        "header_chars": header_chars,
    }
    return stored_text(content, search_text)


def test_search_text_is_header_then_body_and_verifies_as_the_body():
    chunks = ch.sentence_chunks(MESSY, title="Report")
    assert all(_hydrated(c) == c["text"] for c in chunks)
    assert ch.header_search_text("Report › 結果", "body") == (
        "Report › 結果\nbody",
        len("Report › 結果\n"),
    )
    assert ch.header_search_text("", "body") == ("body", 0)


class _RecordingEmbedder:
    client = SimpleNamespace(tokenizer=None)

    def __init__(self):
        self.inputs: list[str] = []

    def embed_documents(self, texts):
        self.inputs.extend(texts)
        return [[float(len(t)), 1.0] for t in texts]


def test_sent_v1_embeds_passage_header_newline_body():
    embedder = _RecordingEmbedder()
    text = f"# Paper\n\n{_prose(40)}\n\n## Methods\n\n{_prose(40, first=50)}"
    (result,) = ew.chunk_and_embed(
        embedder, [text], prefix_model_name=E5, max_tokens=256, scheme="sent-v1", titles=["Paper"]
    )
    chunks = result["chunks"]
    assert result["scheme"] == "sent-v1" and len(result["vectors"]) == len(chunks)
    assert embedder.inputs == [f"passage: {c['header']}\n{c['text']}" for c in chunks]
    assert {"header", "section", "num_tokens"} <= set(chunks[0])


def test_a_request_without_a_scheme_keeps_fixed_v0(monkeypatch):
    embedder = _RecordingEmbedder()
    monkeypatch.setattr(ew, "_load_embedder", lambda _d: embedder)
    request = {"model_dir": "/model", "prefix_model_name": E5, "texts": ["# T\n\nOne. Two."]}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request)))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    assert ew.main() == 0
    (result,) = json.loads(out.getvalue())["results"]
    assert "scheme" not in result and set(result["chunks"][0]) == {"text", "start", "end"}
    assert embedder.inputs == ["passage: # T\n\nOne. Two."]  # body only, no header


def test_the_worker_request_carries_scheme_and_titles(monkeypatch):
    embedder = _RecordingEmbedder()
    monkeypatch.setattr(ew, "_load_embedder", lambda _d: embedder)
    request = {
        "model_dir": "/m",
        "prefix_model_name": E5,
        "texts": ["Body one."],
        "scheme": "sent-v1",
        "titles": ["Doc"],
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request)))
    monkeypatch.setattr("sys.stdout", io.StringIO())
    assert ew.main() == 0
    assert embedder.inputs == ["passage: Doc\nBody one."]


# ── scheme selection ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "scheme"), [(None, "fixed-v0"), ("", "fixed-v0"), (" SENT-V1 ", "sent-v1")]
)
def test_scheme_names_resolve(value, scheme):
    assert ch.resolve_scheme(value) == scheme


def test_an_unknown_scheme_is_refused_by_name(monkeypatch):
    with pytest.raises(ValueError, match="unknown chunk scheme 'sent-v2'"):
        ch.resolve_scheme("sent-v2")
    monkeypatch.setenv(ch.CHUNK_SCHEME_ENV, "bogus")
    with pytest.raises(ValueError, match=ch.CHUNK_SCHEME_ENV):
        ch.configured_scheme()
    monkeypatch.delenv(ch.CHUNK_SCHEME_ENV)
    assert ch.configured_scheme() == ch.FIXED_V0  # the default until the S7 gate
