"""Tests for B3 — Result Comparison & Diff."""

import pytest

from umrs.shared.result_diff import (
    DiffEntry,
    DiffReport,
    DiffType,
    ImageDiff,
    NumericDiff,
    ResultComparator,
    TextDiff,
)


class TestNumericDiff:
    def test_compare_scalars_identical(self):
        diff = NumericDiff()
        entry = diff.compare_scalars("x", 1.0, 1.0)
        assert entry.delta == 0.0
        assert entry.similarity == 1.0

    def test_compare_scalars_different(self):
        diff = NumericDiff()
        entry = diff.compare_scalars("x", 1.0, 2.0)
        assert entry.delta == pytest.approx(1.0)
        assert entry.similarity < 1.0

    def test_compare_scalars_zero(self):
        diff = NumericDiff()
        entry = diff.compare_scalars("x", 0.0, 0.0)
        assert entry.similarity == 1.0

    def test_compare_vectors_identical(self):
        diff = NumericDiff()
        v = [1.0, 2.0, 3.0]
        entry = diff.compare_vectors("v", v, v)
        assert entry.similarity == pytest.approx(1.0)

    def test_compare_vectors_different(self):
        diff = NumericDiff()
        entry = diff.compare_vectors("v", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert entry.similarity < 1.0
        assert entry.similarity >= 0.0

    def test_compare_vectors_length_mismatch(self):
        diff = NumericDiff()
        entry = diff.compare_vectors("v", [1.0], [1.0, 2.0])
        assert entry.similarity == 0.0

    def test_compare_dicts(self):
        diff = NumericDiff()
        entries = diff.compare_dicts("", {"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 3.0})
        assert len(entries) == 2
        a_entry = [e for e in entries if e.key == "a"][0]
        assert a_entry.similarity == 1.0


class TestTextDiff:
    def test_identical(self):
        diff = TextDiff()
        entry = diff.compare("t", "hello", "hello")
        assert entry.similarity == 1.0

    def test_different(self):
        diff = TextDiff()
        entry = diff.compare("t", "hello", "world")
        assert entry.similarity < 1.0

    def test_empty(self):
        diff = TextDiff()
        entry = diff.compare("t", "", "")
        assert entry.similarity == 1.0


class TestImageDiff:
    def test_identical_flat(self):
        diff = ImageDiff()
        pixels = [128] * 100
        entry = diff.compare_pixels("img", pixels, pixels)
        assert entry.similarity == 1.0

    def test_different_pixels(self):
        diff = ImageDiff()
        p1 = [0] * 100
        p2 = [255] * 100
        entry = diff.compare_pixels("img", p1, p2)
        assert entry.similarity < 1.0

    def test_size_mismatch(self):
        diff = ImageDiff()
        entry = diff.compare_pixels("img", [0] * 10, [0] * 20)
        assert entry.similarity == 0.0


class TestDiffReport:
    def test_report_properties(self):
        entries = [
            DiffEntry(key="a", diff_type=DiffType.NUMERIC, value_a=1, value_b=1, delta=0, similarity=1.0),
            DiffEntry(key="b", diff_type=DiffType.NUMERIC, value_a=1, value_b=2, delta=1, similarity=0.5),
        ]
        report = DiffReport(label_a="A", label_b="B", entries=entries)
        assert report.total_comparisons == 2
        assert report.identical_count == 1
        assert report.changed_count == 1
        assert 0 < report.overall_similarity < 1.0

    def test_summary(self):
        entries = [
            DiffEntry(key="x", diff_type=DiffType.TEXT, value_a="a", value_b="a", similarity=1.0),
        ]
        report = DiffReport(label_a="A", label_b="B", entries=entries)
        s = report.summary()
        assert "1" in s  # total comparisons


class TestResultComparator:
    def test_compare_flat_dicts(self):
        comp = ResultComparator()
        report = comp.compare(
            {"score": 0.95, "label": "cat"},
            {"score": 0.90, "label": "cat"},
        )
        assert report.total_comparisons >= 2
        assert report.changed_count >= 1  # score differs

    def test_compare_nested_dicts(self):
        comp = ResultComparator()
        a = {"results": {"accuracy": 0.9, "loss": 0.1}}
        b = {"results": {"accuracy": 0.9, "loss": 0.2}}
        report = comp.compare(a, b)
        assert report.total_comparisons >= 2

    def test_compare_with_lists(self):
        comp = ResultComparator()
        a = {"embeddings": [1.0, 2.0, 3.0]}
        b = {"embeddings": [1.0, 2.0, 3.0]}
        report = comp.compare(a, b)
        assert report.identical_count >= 1

    def test_compare_empty(self):
        comp = ResultComparator()
        report = comp.compare({}, {})
        assert report.total_comparisons == 0
