"""Blank Project fixture helper for RC1 E2E tests (Demo 08 mix)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_PLATFORM_ROOT = Path(__file__).resolve().parents[3]
_DEMO_FIXTURE = _PLATFORM_ROOT / "demos" / "08_scientific_data_management" / "fixture"


def _minimal_pdf_bytes(title: str) -> bytes:
    body = f"%PDF-1.4\n% {title}\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
    return body.encode("utf-8")


def materialize_blank_project(target: Path, *, paper_count: int = 15) -> Path:
    """Create the RC1 Blank Project artifact mix under ``target``."""
    target.mkdir(parents=True, exist_ok=True)

    if _DEMO_FIXTURE.is_dir():
        for name in ("notebooks", "code", "data", "README.md", "MANIFEST.json"):
            src = _DEMO_FIXTURE / name
            if not src.exists():
                continue
            dest = target / name
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest)
    else:
        (target / "README.md").write_text(
            "# Blank Project\n\nGCMC MOF yield research fixture.\n",
            encoding="utf-8",
        )
        notebooks = target / "notebooks"
        notebooks.mkdir(parents=True, exist_ok=True)
        nb = {
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": {"kernelspec": {"name": "python3", "language": "python"}},
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": "# GCMC yield analysis",
                    "metadata": {},
                },
                {
                    "cell_type": "code",
                    "source": "import pandas as pd",
                    "execution_count": 1,
                    "outputs": [],
                    "metadata": {},
                },
            ],
        }
        (notebooks / "analysis.ipynb").write_text(json.dumps(nb), encoding="utf-8")
        (notebooks / "preprocessing.ipynb").write_text(json.dumps(nb), encoding="utf-8")

        code_dir = target / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "model.py").write_text(
            '"""GCMC MOF yield model."""\n\ndef predict_yield(x):\n    return x\n',
            encoding="utf-8",
        )

        data_dir = target / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "batch-3-yields.csv").write_text(
            "mof_id,yield_pct\nMOF-1,12.5\n",
            encoding="utf-8",
        )

    papers_dir = target / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, paper_count + 1):
        paper_path = papers_dir / f"gcmc-mof-paper-{index:02d}.pdf"
        if not paper_path.is_file():
            paper_path.write_bytes(
                _minimal_pdf_bytes(f"GCMC MOF yield study paper {index:02d}")
            )

    return target
