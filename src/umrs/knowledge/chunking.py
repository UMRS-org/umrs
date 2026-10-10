"""Chunk schemes of the knowledge embed lane (ADR-093 D-11, C9).

A site picks one with ``UMRS_KNOWLEDGE_CHUNK_SCHEME``:

- ``fixed-v0`` (the default until the S7 gate passes): windows of
  ``max_tokens`` model tokens, no header — ``shared/utils._chunk_text_without_nlp``,
  unchanged.
- ``sent-v1``: sentence-aware chunks.

  * A deterministic regex splits on sentence ends (``. ! ?`` with an
    abbreviation guard; ``。 ！ ？ ；``), markdown headings, blank lines and
    list items.
  * Sentences pack to a target of 200 tokens, at most 256. A body of 384
    tokens or less stays one chunk. The previous chunk's last sentence
    overlaps into the next when it is 48 tokens or less; a tail under 48
    tokens merges back when the merged chunk stays within 256; a sentence
    over the maximum falls back to token windows.
  * The header title is the artifact's metadata title, else the level-1
    heading opening a markdown text, else its source name
    (:func:`pick_title`); in code and config ``#`` lines are comments.
  * Every chunk is ``text[start:end]`` of the original text, so its md5
    verifies against the artifact (hydrate, backfill, export).
  * A header ``{title} › {section}`` (at most 32 tokens, no LLM) is embedded
    with the body (``passage: `` + header + newline + body) and stored before
    it in ``search_text``; ``content.header_chars`` says where the body
    starts (:func:`header_search_text`).

This runs in the embed worker / sandbox lane (CONSTITUTION D-12), never in
the server. Token counts come from the model's own tokenizer when it is
given, else from a deterministic approximation.
"""

from __future__ import annotations

import os
import re
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Any

from umrs.shared.utils import _chunk_text_without_nlp, compute_md5

FIXED_V0 = "fixed-v0"
SENT_V1 = "sent-v1"
SCHEMES = (FIXED_V0, SENT_V1)
CHUNK_SCHEME_ENV = "UMRS_KNOWLEDGE_CHUNK_SCHEME"

TARGET_TOKENS = 200
MAX_TOKENS = 256
ONE_CHUNK_TOKENS = 384
SMALL_TOKENS = 48  # the overlap ceiling and the tail floor
HEADER_TOKENS = 32
SECTION_TOKENS = 12  # the section's share of the header when both fit badly
HEADER_SEP = " › "


def resolve_scheme(value: str | None) -> str:
    """The named scheme; empty means ``fixed-v0``; anything else is refused."""
    scheme = str(value or "").strip().lower() or FIXED_V0
    if scheme not in SCHEMES:
        raise ValueError(
            f"unknown chunk scheme {value!r}: expected one of {', '.join(SCHEMES)}"
        )
    return scheme


def configured_scheme() -> str:
    """This site's scheme (``UMRS_KNOWLEDGE_CHUNK_SCHEME``, default fixed-v0)."""
    try:
        return resolve_scheme(os.environ.get(CHUNK_SCHEME_ENV))
    except ValueError as exc:
        raise ValueError(f"{CHUNK_SCHEME_ENV}: {exc}") from exc


# ── tokens ──────────────────────────────────────────────────────────────────

_CJK = "　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯"
_CJK_CHAR = re.compile(rf"[{_CJK}]")
# Without a tokenizer: one token per CJK character, word run or punctuation mark.
_APPROX_TOKEN = re.compile(rf"[{_CJK}]|[^\W{_CJK}]+|[^\w\s]")


def _token_spans(text: str, tokenizer: Any) -> list[tuple[int, int]]:
    if tokenizer is None:
        return [m.span() for m in _APPROX_TOKEN.finditer(text)]
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return [(int(a), int(b)) for a, b in encoding["offset_mapping"] if b > a]


class TokenIndex:
    """Token boundaries of one text; a range counts the tokens starting in it."""

    def __init__(self, text: str, tokenizer: Any = None) -> None:
        self.spans = _token_spans(text, tokenizer)
        self._starts = [start for start, _ in self.spans]

    def __len__(self) -> int:
        return len(self.spans)

    def count(self, start: int, end: int) -> int:
        return bisect_left(self._starts, end) - bisect_left(self._starts, start)

    def windows(self, start: int, end: int, size: int) -> list[tuple[int, int]]:
        """``[start, end)`` cut into runs of ``size`` tokens."""
        lo, hi = bisect_left(self._starts, start), bisect_left(self._starts, end)
        return [
            (self.spans[i][0], min(self.spans[min(i + size, hi) - 1][1], end))
            for i in range(lo, hi, size)
        ]


# ── headers and titles ──────────────────────────────────────────────────────

_HEADING_LINE = r"[ \t]{0,3}#{1,6}[ \t]+(?P<title>\S.*?)(?:[ \t]+#+)?[ \t]*"
_HEADING = re.compile(_HEADING_LINE)
_FIRST_HEADING = re.compile(rf"^{_HEADING_LINE}$", re.MULTILINE)
_H1 = re.compile(r"[ \t]{0,3}#[ \t]+(?P<title>\S.*?)(?:[ \t]+#+)?[ \t]*")
_NON_BLANK = re.compile(r"\S")
# '#' starts a heading in markdown (and in an unnamed text); in code, shell,
# YAML or a plain text file it is a comment, never a title (review
# chunk-data-8, 2026-10-10).
_HEADING_SUFFIXES = frozenset(
    {"", ".md", ".markdown", ".mdown", ".mkd", ".mdx", ".qmd", ".rmd"}
)


def clean_text(value: Any) -> str:
    """One line, single spaces."""
    return " ".join(str(value or "").split())


def first_heading(text: str) -> str:
    """The text of the first markdown heading, or ``""``."""
    match = _FIRST_HEADING.search(text)
    return clean_text(match.group("title")) if match else ""


def has_heading_syntax(source_name: str) -> bool:
    """Whether the artifact's '#' lines are markdown headings: a markdown
    file, or a text without a suffix — not code, config or plain text."""
    return PurePosixPath(source_name).suffix.lower() in _HEADING_SUFFIXES


def leading_title(text: str) -> str:
    """The text of the level-1 heading on the text's first non-blank line,
    or ``""`` (a later heading names a section, not the document)."""
    first = _NON_BLANK.search(text)
    if first is None:
        return ""
    start = text.rfind("\n", 0, first.start()) + 1
    end = text.find("\n", first.start())
    line = text[start : end if end >= 0 else len(text)].rstrip("\r")
    match = _H1.fullmatch(line)
    return clean_text(match["title"]) if match else ""


def pick_title(meta_title: str, text: str, source_name: str) -> str:
    """The header title: the artifact's metadata title, else the level-1
    heading that opens a markdown text, else the source name (the file name
    tells chunks apart better than a comment or a later section heading)."""
    heading = leading_title(text) if has_heading_syntax(source_name) else ""
    return clean_text(meta_title) or heading or clean_text(source_name)


def _same(a: str, b: str) -> bool:
    return a.casefold() == b.casefold()


def clip_tokens(text: str, limit: int, tokenizer: Any = None) -> str:
    """``text`` cut after its ``limit``-th token."""
    if limit <= 0:
        return ""
    index = TokenIndex(text, tokenizer)
    if len(index) <= limit:
        return text
    return text[: index.spans[limit - 1][1]].rstrip()


def build_header(title: str, section: str, tokenizer: Any = None) -> str:
    """``{title} › {section}`` within ``HEADER_TOKENS``; either part alone
    when the other is empty or the same."""
    title, section = clean_text(title), clean_text(section)
    if not section or _same(section, title):
        return clip_tokens(title, HEADER_TOKENS, tokenizer)
    if not title:
        return clip_tokens(section, HEADER_TOKENS, tokenizer)
    section = clip_tokens(section, SECTION_TOKENS, tokenizer)
    room = HEADER_TOKENS - len(TokenIndex(HEADER_SEP + section, tokenizer))
    header = f"{clip_tokens(title, room, tokenizer)}{HEADER_SEP}{section}"
    return clip_tokens(header, HEADER_TOKENS, tokenizer)


def header_search_text(header: str, body: str) -> tuple[str, int]:
    """``(search_text, header_chars)``: the header, a newline, the body.
    ``search_text[header_chars:] == body``, so its md5 is the chunk's."""
    if not header:
        return body, 0
    prefix = f"{header}\n"
    return prefix + body, len(prefix)


def search_text_of(content: dict[str, Any], body: str) -> str | None:
    """The ``search_text`` a chunk node stores for its verified ``body``: the
    body, or ``content.header`` + newline + body when the node records
    ``header_chars`` (sent-v1). ``None`` when the node's header and
    ``header_chars`` disagree — writing the body alone would make hydrate's
    ``search_text[header_chars:]`` miss it. Every writer of the column after
    the embed job (backfill, bundle import) uses this rule."""
    header_chars = content.get("header_chars")
    if not isinstance(header_chars, int) or header_chars <= 0:
        return body
    text, chars = header_search_text(str(content.get("header") or ""), body)
    return text if chars == header_chars else None


def embed_input(chunk: dict[str, Any]) -> str:
    """What the model embeds (before the model's ``passage: `` prefix)."""
    header = str(chunk.get("header") or "")
    return f"{header}\n{chunk['text']}" if header else str(chunk["text"])


# ── blocks: headings, paragraphs, list items ────────────────────────────────

_LIST_ITEM = re.compile(r"[ \t]*(?:[-*+•]|\d{1,3}[.)])[ \t]+\S")


@dataclass(frozen=True)
class _Block:
    start: int
    end: int
    heading: str | None = None


class _BlockScanner:
    """Lines into blocks. A blank line, a heading or a list item ends the
    open paragraph; a heading is a block of its own. Without ``headings``
    (code, config: '#' is a comment there) no line is a heading."""

    def __init__(self, headings: bool = True) -> None:
        self.blocks: list[_Block] = []
        self._open: list[int] | None = None
        self._headings = headings

    def flush(self) -> None:
        if self._open is not None:
            self.blocks.append(_Block(self._open[0], self._open[1]))
            self._open = None

    def line(self, start: int, body: str) -> None:
        end = start + len(body)
        heading = _HEADING.fullmatch(body) if self._headings else None
        if heading or not body.strip() or _LIST_ITEM.match(body):
            self.flush()
        if heading:
            self.blocks.append(_Block(start, end, clean_text(heading["title"])))
        elif body.strip():
            self._extend(start, end)

    def _extend(self, start: int, end: int) -> None:
        if self._open is None:
            self._open = [start, end]
        else:
            self._open[1] = end


def _blocks(text: str, headings: bool = True) -> list[_Block]:
    scanner = _BlockScanner(headings)
    position = 0
    for line in text.split("\n"):
        scanner.line(position, line.rstrip("\r"))
        position += len(line) + 1
    scanner.flush()
    return scanner.blocks


# ── sentences ───────────────────────────────────────────────────────────────

_CJK_ENDS = "。！？；"
_TERMINATOR = re.compile(rf"[{_CJK_ENDS}]|[.!?]+[\"'”’)\]]*(?=\s)")
_NEXT_START = re.compile(r"\s+[\"'“‘(\[]*(\S)")
_INITIALISM = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]")  # e.g, i.e, U.S
_WORD_BEFORE = re.compile(r"\S*$")
_ABBREVIATIONS = frozenset(
    {
        "al", "approx", "ca", "cf", "ch", "co", "corp", "dept", "dr", "ed",
        "eds", "eq", "eqs", "est", "etc", "fig", "figs", "inc", "incl", "jr",
        "ltd", "mr", "mrs", "ms", "mt", "prof", "ref", "refs", "resp", "sec",
        "sect", "sr", "st", "suppl", "tab", "univ", "vol", "vols", "vs",
    }
)  # fmt: skip
# Abbreviations only before a number ("No. 5", "pp. 12"): "said no. Next" ends.
_NUMBER_ABBREVIATIONS = frozenset({"no", "nos", "p", "pp"})


def _word_before(text: str, position: int) -> str:
    window = text[max(0, position - 24) : position]
    run = _WORD_BEFORE.search(window)
    return (run.group() if run else "").lstrip("([{\"'“‘")


def _is_abbreviation(word: str, following: str) -> bool:
    """Does the ``.`` after ``word`` close an abbreviation (``following`` is
    the next sentence's first character)?"""
    lowered = word.lower()
    if lowered in _NUMBER_ABBREVIATIONS:
        return following.isdigit()
    initial = len(word) == 1 and word.isupper()  # "J. Smith"
    return initial or lowered in _ABBREVIATIONS or bool(_INITIALISM.fullmatch(word))


def _starts_sentence(char: str) -> bool:
    return char.isupper() or char.isdigit() or bool(_CJK_CHAR.match(char))


def _is_boundary(text: str, match: re.Match[str], end: int) -> bool:
    """A sentence ends at ``match``: a CJK end mark, or ``. ! ?`` followed by
    a capital, a digit or CJK text and not closing an abbreviation."""
    mark = match.group()
    if mark[0] in _CJK_ENDS:
        return True
    following = _NEXT_START.match(text, match.end(), end)
    if following is None or not _starts_sentence(following.group(1)):
        return False
    word = _word_before(text, match.start())
    return mark[0] != "." or not _is_abbreviation(word, following.group(1))


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    cuts = [start]
    cuts += [
        m.end()
        for m in _TERMINATOR.finditer(text, start, end)
        if _is_boundary(text, m, end)
    ]
    cuts.append(end)
    return [span for a, b in pairwise(cuts) if (span := _trim(text, a, b))]


# ── segments and packing ────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Segment:
    start: int
    end: int
    tokens: int
    section: str
    heading: bool = False
    window: bool = False


def _block_spans(text: str, block: _Block) -> list[tuple[int, int]]:
    if block.heading is None:
        return _sentences(text, block.start, block.end)
    span = _trim(text, block.start, block.end)
    return [span] if span else []


def sentence_spans(text: str, headings: bool = True) -> list[tuple[int, int]]:
    """``(start, end)`` of every sentence, heading and list item of ``text``
    in order — the units sent-v1 packs (before oversize windowing)."""
    blocks = _blocks(text, headings)
    return [span for block in blocks for span in _block_spans(text, block)]


def _sized(segment: _Segment, index: TokenIndex) -> list[_Segment]:
    """The segment, or token windows of it when it is over the maximum."""
    if segment.tokens <= MAX_TOKENS:
        return [segment]
    return [
        _Segment(a, b, index.count(a, b), segment.section, segment.heading, window=True)
        for a, b in index.windows(segment.start, segment.end, MAX_TOKENS)
    ]


def _segments(text: str, index: TokenIndex, headings: bool = True) -> list[_Segment]:
    out: list[_Segment] = []
    section = ""
    for block in _blocks(text, headings):
        section = block.heading if block.heading is not None else section
        for start, end in _block_spans(text, block):
            segment = _Segment(
                start, end, index.count(start, end), section, block.heading is not None
            )
            out.extend(_sized(segment, index))
    return out


def _closes(total: int, own: int, segment: _Segment) -> bool:
    """Does ``segment`` start a new chunk after one of ``total`` tokens,
    ``own`` of them not overlap? A segment that would take the chunk past
    the maximum always does (a short opening sentence carried a long one to
    290 tokens before, review chunk-data-7); otherwise a chunk whose own
    text is under the tail floor never closes, and a heading starts one."""
    if own and total + segment.tokens > MAX_TOKENS:
        return True
    if own < SMALL_TOKENS:
        return False
    return segment.heading or total >= TARGET_TOKENS


def _overlaps(segments: Sequence[_Segment], first: int, at: int) -> bool:
    """Does the chunk starting at ``at`` repeat the last sentence of the one
    that spans ``first .. at - 1``?"""
    last, nxt = segments[at - 1], segments[at]
    return (
        at - 1 > first
        and not (last.heading or last.window or nxt.heading)
        and last.tokens <= SMALL_TOKENS
        and last.tokens + nxt.tokens <= MAX_TOKENS
    )


def _own_tokens(segments: Sequence[_Segment], group: list[int]) -> int:
    return sum(s.tokens for s in segments[group[1] : group[2] + 1])


def _pack(segments: Sequence[_Segment]) -> list[list[int]]:
    """``[first, own, last]`` segment indices per chunk; ``own`` is the first
    segment that is not overlap."""
    groups: list[list[int]] = []
    first = own = total = own_total = 0
    for at, segment in enumerate(segments):
        if _closes(total, own_total, segment):
            groups.append([first, own, at - 1])
            first, own = (at - 1 if _overlaps(segments, first, at) else at), at
            total, own_total = sum(s.tokens for s in segments[first:at]), 0
        total += segment.tokens
        own_total += segment.tokens
    groups.append([first, own, len(segments) - 1])
    if _merges_back(segments, groups):
        tail = groups.pop()
        groups[-1][2] = tail[2]  # the tail merges back
    return groups


def _merges_back(segments: Sequence[_Segment], groups: list[list[int]]) -> bool:
    """A tail under the floor merges into the chunk before it — unless the
    merged chunk would pass the maximum."""
    if len(groups) < 2 or _own_tokens(segments, groups[-1]) >= SMALL_TOKENS:
        return False
    merged = sum(s.tokens for s in segments[groups[-2][0] : groups[-1][2] + 1])
    return merged <= MAX_TOKENS


def _section_of(segments: Sequence[_Segment]) -> str:
    """The section of the chunk's first sentence (its last heading if it
    holds only headings)."""
    for segment in segments:
        if not segment.heading:
            return segment.section
    return segments[-1].section if segments else ""


def _chunk(
    text: str,
    index: TokenIndex,
    segments: Sequence[_Segment],
    group: list[int],
    title: str,
    tokenizer: Any,
) -> dict[str, Any]:
    first, own, last = group
    start, end = segments[first].start, segments[last].end
    body = text[start:end]
    section = _section_of(segments[own : last + 1])
    section = "" if _same(section, clean_text(title)) else section
    return {
        "text": body,
        "start": start,
        "end": end,
        "num_tokens": index.count(start, end),
        "md5": compute_md5(body),
        "header": build_header(title, section, tokenizer),
        "section": section,
    }


def sentence_chunks(
    text: str, *, title: str = "", tokenizer: Any = None, headings: bool = True
) -> list[dict[str, Any]]:
    """``sent-v1`` chunks of ``text``: ``{text, start, end, num_tokens, md5,
    header, section}`` with ``text == text[start:end]``. ``headings=False``
    (code, config: :func:`has_heading_syntax`) reads no '#' line as a
    heading."""
    index = TokenIndex(text, tokenizer)
    segments = _segments(text, index, headings)
    if not segments:
        return []
    one = len(index) <= ONE_CHUNK_TOKENS
    groups = [[0, 0, len(segments) - 1]] if one else _pack(segments)
    return [_chunk(text, index, segments, g, title, tokenizer) for g in groups]


def chunk_text(
    text: str,
    *,
    scheme: str = FIXED_V0,
    tokenizer: Any = None,
    max_tokens: int | None = None,
    title: str = "",
    headings: bool = True,
) -> list[dict[str, Any]]:
    """Chunks of ``text`` under ``scheme`` (``max_tokens`` is fixed-v0's window;
    ``title`` and ``headings`` are sent-v1's header title and heading rule)."""
    if resolve_scheme(scheme) == SENT_V1:
        return sentence_chunks(
            text, title=title, tokenizer=tokenizer, headings=headings
        )
    return _chunk_text_without_nlp(text, tokenizer=tokenizer, max_tokens=max_tokens)


__all__ = [
    "CHUNK_SCHEME_ENV",
    "FIXED_V0",
    "HEADER_TOKENS",
    "MAX_TOKENS",
    "ONE_CHUNK_TOKENS",
    "SCHEMES",
    "SENT_V1",
    "SMALL_TOKENS",
    "TARGET_TOKENS",
    "TokenIndex",
    "build_header",
    "chunk_text",
    "clip_tokens",
    "configured_scheme",
    "embed_input",
    "first_heading",
    "has_heading_syntax",
    "header_search_text",
    "leading_title",
    "pick_title",
    "resolve_scheme",
    "search_text_of",
    "sentence_chunks",
    "sentence_spans",
]
