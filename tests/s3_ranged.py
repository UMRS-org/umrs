"""A GET as S3 answers it — ranged when asked — for the in-memory S3 fakes.

Cycle-3 P1 (2026-10-22): the knowledge chain reads only the bytes it uses (a
ranged GET for understand, embed and retrieval) and copies staged uploads
server side, so the fakes answer ``Range`` and ``copy_object`` like a store.
"""

from __future__ import annotations

import io
from typing import Any


def ranged(body: bytes, range_header: str = "") -> dict[str, Any]:
    """``get_object``'s answer for *body*, honouring ``Range: bytes=a-b``."""
    if not range_header:
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}
    first, _, last = range_header.removeprefix("bytes=").partition("-")
    start = int(first)
    part = body[start : int(last) + 1]
    return {
        "Body": io.BytesIO(part),
        "ContentLength": len(part),
        "ContentRange": f"bytes {start}-{start + len(part) - 1}/{len(body)}",
    }
