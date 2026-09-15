"""Tests for modality detection and text extraction."""

import json

from umrs.data_catalog.extract import extract_content
from umrs.data_catalog.modality import detect_modality, filename_signals


def test_detect_modality_text():
    assert detect_modality("notes.txt", b"hello world") == "text"


def test_detect_modality_json():
    assert detect_modality("data.json", b'{"a": 1}') == "structured"


def test_detect_modality_pdf_magic():
    assert detect_modality("doc.pdf", b"%PDF-1.4\n") == "document"


def test_filename_signals():
    signals = filename_signals("proj/run_01_results.csv")
    assert "run_01_results.csv" in signals
    assert "run" in signals or "01" in signals


def test_extract_json_summary():
    payload = {"experiment": "alpha", "value": 42}
    summary, text, tags = extract_content("out.json", json.dumps(payload).encode())
    assert "alpha" in summary or "experiment" in text
    assert "experiment" in tags or summary
