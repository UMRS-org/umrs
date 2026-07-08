"""Tests for CSV value sampling into catalog tags."""

from umrs.data_catalog.extract import extract_content


def test_extract_csv_includes_sample_values_as_tags():
    body = b"batch_id,yield_pct\nbatch-1,72.1\nbatch-3,85.0\n"
    _, _, tags = extract_content("batch-yields.csv", body)
    assert "batch_id" in tags
    assert "yield_pct" in tags
    assert "batch-1" in tags or "batch-3" in tags
