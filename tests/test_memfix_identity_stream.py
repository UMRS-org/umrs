"""Cycle-3 memfix #4 (2026-10-22): the streamed canonical identity of a long
line with no ASCII pair (CJK prose without line breaks) is linear in time
and bounded in memory.

Found by the mem review (``.stab-cycle3/memreview/memreview-v3-0``):
``IdentityStream`` flushed a long open line only between two ASCII
characters. Without one, it kept the whole line and re-joined, re-split and
re-scanned it for every 1 MiB part: 400 MiB of single-line Japanese took 62 s
and +800 MB RSS (ASCII: 1.5 s), on the UMRS event loop. Now the line is
flushed before any stable starter and is held as parts joined once.
"""

from __future__ import annotations

import hashlib
import random
import tracemalloc
import unicodedata

import pytest
from umrs.knowledge import canonical_identity as ci

# Pieces NFC composes, reorders or decomposes around, and starters a flush
# may or may not cut before.
_PIECES = [
    "a",
    " ",
    "\t",
    "\r",
    "\n",
    "\r\n",
    "é",
    "́",
    "̣",
    "é",
    "Å",
    "ᄀ",  # Hangul L
    "ᅡ",  # V
    "ᆨ",  # T
    "가",  # LV syllable (+ T composes)
    "ெ",  # Tamil vowel sign E (+ AA composes; both starters)
    "ா",
    "ཱི",  # a starter decomposing into non-starters
    "̈́",
    "か",  # か (+ U+3099 composes)
    "゙",
    "ｶﾞ",
    "豈",  # a compatibility ideograph (a singleton)
    "क़",  # a composition exclusion
    "日本語",
    "、",
    "　",
    "\U0001f600",
    "\U0002f800",
]
_RAW = [b"\x00", b"\xff", b"\xef\xbb\xbf", b"\xe6\x97"]


def _sample(rng: random.Random) -> bytes:
    parts = [
        rng.choice(_PIECES).encode() if rng.random() < 0.97 else rng.choice(_RAW)
        for _ in range(rng.randint(0, 40))
    ]
    return b"".join(parts)


def _streamed(raw: bytes, rng: random.Random, most: int = 9) -> ci.CanonicalIdentity:
    stream = ci.IdentityStream()
    i = 0
    while i < len(raw):
        size = rng.randint(1, most)
        stream.feed(raw[i : i + size])
        i += size
    return stream.identity()


@pytest.mark.parametrize(("flush", "scan"), [(1, 4096), (3, 4096), (2, 1), (6, 2)])
def test_the_streamed_identity_is_the_identity_around_every_starter(monkeypatch, flush, scan):
    monkeypatch.setattr(ci, "_FLUSH_CHARS", flush)
    monkeypatch.setattr(ci, "_FLUSH_SCAN_CHARS", scan, raising=False)
    rng = random.Random(20261022 + flush * 7 + scan)
    for _ in range(4000):
        raw = _sample(rng)
        assert _streamed(raw, rng) == ci.identity_of(raw, "p"), raw


def _composed_seconds() -> list[str]:
    """The second character of every BMP character NFC keeps composed."""
    pairs = (unicodedata.normalize("NFD", chr(cp)) for cp in range(0x80, 0x10000))
    return [d[1] for d in pairs if len(d) == 2 and unicodedata.normalize("NFC", d) != d]


def test_no_character_a_flush_cuts_before_is_composed_onto_its_predecessor():
    """Every canonical composition's second character is refused as a cut
    point (the BMP; Hangul is algorithmic and checked by name)."""
    seconds = _composed_seconds()
    assert len(seconds) > 900
    assert [c for c in seconds if ci._stable_starter(c)] == []
    refused = ("\u1161", "\u1175", "\u11a8", "\u11c2", " ", "\t", "\u0301")
    assert list(map(ci._stable_starter, refused)) == [False] * len(refused)
    assert list(map(ci._stable_starter, ("\u65e5", "\uac00", "a"))) == [True] * 3


def _cjk_line(mib: int) -> bytes:
    unit = "日本語の文章、".encode()
    return unit * (mib * (1 << 20) // len(unit))


def test_a_single_line_cjk_text_streams_in_bounded_memory(monkeypatch):
    part = 64 * 1024
    monkeypatch.setattr(ci, "_FLUSH_CHARS", part)
    raw = _cjk_line(4)
    tracemalloc.start()
    try:
        stream = ci.IdentityStream()
        for i in range(0, len(raw), part):
            stream.feed(raw[i : i + part])
        got = stream.identity()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert got == ci.identity_of(raw, hashlib.sha256(raw).hexdigest())
    # The open line was held whole (4 MiB of UTF-8, ~2.8 MB as str) and
    # copied once per part; now about two parts are.
    assert peak < 1_500_000, peak
