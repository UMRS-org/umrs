"""
B3 — Result Comparison & Diff

Compare results across pipeline runs, experiments, or model versions.
Supports numeric (embeddings, scores), image (SSIM/MSE), and text
(edit distance) comparisons.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DiffType(str, Enum):
    """Type of comparison performed."""

    NUMERIC = "numeric"
    TEXT = "text"
    IMAGE = "image"
    STRUCTURAL = "structural"


@dataclass
class DiffEntry:
    """A single difference between two values."""

    key: str
    diff_type: DiffType
    value_a: Any
    value_b: Any
    delta: float | None = None
    similarity: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_identical(self) -> bool:
        if self.similarity is not None:
            return self.similarity >= 1.0
        return self.value_a == self.value_b


@dataclass
class DiffReport:
    """Structured comparison report."""

    label_a: str
    label_b: str
    entries: list[DiffEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_comparisons(self) -> int:
        return len(self.entries)

    @property
    def identical_count(self) -> int:
        return sum(1 for e in self.entries if e.is_identical)

    @property
    def changed_count(self) -> int:
        return self.total_comparisons - self.identical_count

    @property
    def overall_similarity(self) -> float:
        if not self.entries:
            return 1.0
        sims = [e.similarity for e in self.entries if e.similarity is not None]
        if not sims:
            identical = sum(1 for e in self.entries if e.is_identical)
            return identical / len(self.entries) if self.entries else 1.0
        return sum(sims) / len(sims)

    def summary(self) -> str:
        lines = [
            f"Diff: {self.label_a} vs {self.label_b}",
            f"  Comparisons: {self.total_comparisons}",
            f"  Identical:   {self.identical_count}",
            f"  Changed:     {self.changed_count}",
            f"  Similarity:  {self.overall_similarity:.4f}",
        ]
        return "\n".join(lines)

    def changed_entries(self) -> list[DiffEntry]:
        return [e for e in self.entries if not e.is_identical]


class NumericDiff:
    """Compare numeric values (scalars, vectors, matrices)."""

    @staticmethod
    def compare_scalars(key: str, a: float, b: float) -> DiffEntry:
        delta = abs(a - b)
        max_val = max(abs(a), abs(b), 1e-10)
        similarity = 1.0 - min(delta / max_val, 1.0)
        return DiffEntry(
            key=key,
            diff_type=DiffType.NUMERIC,
            value_a=a,
            value_b=b,
            delta=delta,
            similarity=similarity,
        )

    @staticmethod
    def compare_vectors(key: str, a: Sequence[float], b: Sequence[float]) -> DiffEntry:
        if len(a) != len(b):
            return DiffEntry(
                key=key,
                diff_type=DiffType.NUMERIC,
                value_a=f"len={len(a)}",
                value_b=f"len={len(b)}",
                similarity=0.0,
                details={"error": "dimension mismatch"},
            )

        cos_sim = NumericDiff._cosine_similarity(a, b)
        mse = sum((x - y) ** 2 for x, y in zip(a, b, strict=False)) / len(a)
        return DiffEntry(
            key=key,
            diff_type=DiffType.NUMERIC,
            value_a=f"vec[{len(a)}]",
            value_b=f"vec[{len(b)}]",
            similarity=cos_sim,
            details={"cosine_similarity": cos_sim, "mse": mse, "dimension": len(a)},
        )

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return max(0.0, min(dot / (norm_a * norm_b), 1.0))

    @staticmethod
    def compare_dicts(
        key_prefix: str,
        a: dict[str, float],
        b: dict[str, float],
    ) -> list[DiffEntry]:
        entries = []
        all_keys = sorted(set(a.keys()) | set(b.keys()))
        for k in all_keys:
            full_key = f"{key_prefix}.{k}" if key_prefix else k
            if k in a and k in b:
                entries.append(NumericDiff.compare_scalars(full_key, a[k], b[k]))
            elif k in a:
                entries.append(
                    DiffEntry(
                        key=full_key,
                        diff_type=DiffType.NUMERIC,
                        value_a=a[k],
                        value_b=None,
                        similarity=0.0,
                        details={"status": "removed"},
                    )
                )
            else:
                entries.append(
                    DiffEntry(
                        key=full_key,
                        diff_type=DiffType.NUMERIC,
                        value_a=None,
                        value_b=b[k],
                        similarity=0.0,
                        details={"status": "added"},
                    )
                )
        return entries


class TextDiff:
    """Compare text values using edit distance and sequence matching."""

    @staticmethod
    def compare(key: str, a: str, b: str) -> DiffEntry:
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        unified = list(
            difflib.unified_diff(
                a.splitlines(keepends=True),
                b.splitlines(keepends=True),
                fromfile="a",
                tofile="b",
                lineterm="",
            )
        )
        return DiffEntry(
            key=key,
            diff_type=DiffType.TEXT,
            value_a=a[:200] + ("..." if len(a) > 200 else ""),
            value_b=b[:200] + ("..." if len(b) > 200 else ""),
            similarity=ratio,
            details={
                "edit_ratio": ratio,
                "len_a": len(a),
                "len_b": len(b),
                "diff_lines": len(unified),
                "unified_diff": "".join(unified[:50]),
            },
        )


class ImageDiff:
    """Compare image data using pixel-level metrics (MSE, PSNR).

    Note: Operates on raw pixel arrays (list of floats, 0–255 range).
    For production use with actual images, convert to numpy arrays first.
    """

    @staticmethod
    def compare_pixels(
        key: str,
        a: Sequence[float],
        b: Sequence[float],
        max_pixel_value: float = 255.0,
    ) -> DiffEntry:
        if len(a) != len(b):
            return DiffEntry(
                key=key,
                diff_type=DiffType.IMAGE,
                value_a=f"pixels={len(a)}",
                value_b=f"pixels={len(b)}",
                similarity=0.0,
                details={"error": "size mismatch"},
            )

        mse = sum((x - y) ** 2 for x, y in zip(a, b, strict=False)) / len(a)
        if mse < 1e-10:
            psnr = float("inf")
            similarity = 1.0
        else:
            psnr = 10.0 * math.log10((max_pixel_value**2) / mse)
            # Normalize PSNR to 0–1 range (60dB is effectively perfect)
            similarity = min(psnr / 60.0, 1.0)

        return DiffEntry(
            key=key,
            diff_type=DiffType.IMAGE,
            value_a=f"pixels={len(a)}",
            value_b=f"pixels={len(b)}",
            similarity=similarity,
            details={"mse": mse, "psnr": psnr, "pixel_count": len(a)},
        )


class ResultComparator:
    """Main comparison engine for UMRS results.

    Usage::

        comparator = ResultComparator()
        report = comparator.compare(
            results_a={"score": 0.95, "embedding": [0.1, 0.2, ...]},
            results_b={"score": 0.93, "embedding": [0.1, 0.19, ...]},
            label_a="run-001",
            label_b="run-002",
        )
        print(report.summary())
    """

    def __init__(self, tolerance: float = 1e-6) -> None:
        self.tolerance = tolerance

    def compare(
        self,
        results_a: dict[str, Any],
        results_b: dict[str, Any],
        label_a: str = "A",
        label_b: str = "B",
    ) -> DiffReport:
        report = DiffReport(label_a=label_a, label_b=label_b)
        all_keys = sorted(set(results_a.keys()) | set(results_b.keys()))

        for key in all_keys:
            val_a = results_a.get(key)
            val_b = results_b.get(key)

            if val_a is None or val_b is None:
                report.entries.append(
                    DiffEntry(
                        key=key,
                        diff_type=DiffType.STRUCTURAL,
                        value_a=val_a,
                        value_b=val_b,
                        similarity=0.0,
                        details={
                            "status": "missing_in_a"
                            if val_a is None
                            else "missing_in_b"
                        },
                    )
                )
                continue

            entries = self._compare_values(key, val_a, val_b)
            report.entries.extend(entries)

        return report

    def _compare_values(self, key: str, a: Any, b: Any) -> list[DiffEntry]:
        if isinstance(a, str) and isinstance(b, str):
            return [TextDiff.compare(key, a, b)]

        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return [NumericDiff.compare_scalars(key, float(a), float(b))]

        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if (
                a
                and b
                and isinstance(a[0], (int, float))
                and isinstance(b[0], (int, float))
            ):
                return [NumericDiff.compare_vectors(key, a, b)]
            # Compare as generic sequences
            return self._compare_sequences(key, a, b)

        if isinstance(a, dict) and isinstance(b, dict):
            if all(isinstance(v, (int, float)) for v in a.values()) and all(
                isinstance(v, (int, float)) for v in b.values()
            ):
                return NumericDiff.compare_dicts(key, a, b)
            return self._compare_dicts(key, a, b)

        # Fallback: string comparison
        return [TextDiff.compare(key, str(a), str(b))]

    def _compare_sequences(self, key: str, a: list, b: list) -> list[DiffEntry]:
        entries = []
        max_len = max(len(a), len(b))
        for i in range(max_len):
            elem_key = f"{key}[{i}]"
            if i < len(a) and i < len(b):
                entries.extend(self._compare_values(elem_key, a[i], b[i]))
            else:
                entries.append(
                    DiffEntry(
                        key=elem_key,
                        diff_type=DiffType.STRUCTURAL,
                        value_a=a[i] if i < len(a) else None,
                        value_b=b[i] if i < len(b) else None,
                        similarity=0.0,
                    )
                )
        return entries

    def _compare_dicts(self, key: str, a: dict, b: dict) -> list[DiffEntry]:
        entries = []
        all_keys = sorted(set(a.keys()) | set(b.keys()))
        for k in all_keys:
            full_key = f"{key}.{k}"
            if k in a and k in b:
                entries.extend(self._compare_values(full_key, a[k], b[k]))
            else:
                entries.append(
                    DiffEntry(
                        key=full_key,
                        diff_type=DiffType.STRUCTURAL,
                        value_a=a.get(k),
                        value_b=b.get(k),
                        similarity=0.0,
                        details={
                            "status": "missing_in_a" if k not in a else "missing_in_b"
                        },
                    )
                )
        return entries
