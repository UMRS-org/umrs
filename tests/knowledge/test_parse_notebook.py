"""Tests for the Jupyter notebook parser."""

from __future__ import annotations

import json

from umrs.knowledge.parse_notebook import parse_notebook


def _make_notebook(cells: list[dict]) -> bytes:
    """Create a minimal .ipynb notebook with the given cells."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "language": "python",
                "display_name": "Python 3",
            },
        },
        "cells": cells,
    }
    return json.dumps(nb).encode("utf-8")


def test_parse_markdown_cell():
    """Test parsing a notebook with a single markdown cell."""
    nb = _make_notebook([
        {
            "cell_type": "markdown",
            "source": "# My Experiment\n\nThis is a test.",
            "metadata": {},
        },
    ])

    result = parse_notebook("test.ipynb", nb)

    assert result.error is None
    assert len(result.cells) == 1
    assert result.cells[0].is_markdown()
    assert "My Experiment" in result.extracted_text
    assert "python" in result.tags
    assert "python3" in result.tags


def test_parse_code_cell_with_output():
    """Test parsing a notebook with a code cell that has output."""
    nb = _make_notebook([
        {
            "cell_type": "code",
            "source": "print('Hello, world!')",
            "execution_count": 1,
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": "Hello, world!\n",
                },
            ],
            "metadata": {},
        },
    ])

    result = parse_notebook("test.ipynb", nb)

    assert result.error is None
    assert len(result.cells) == 1
    assert result.cells[0].is_code()
    assert result.cells[0].execution_count == 1
    assert "Hello, world!" in result.extracted_text
    assert result.cells[0].outputs[0]["output_type"] == "stream"


def test_parse_code_cell_with_error():
    """Test parsing a notebook with a code cell that has an error output."""
    nb = _make_notebook([
        {
            "cell_type": "code",
            "source": "1 / 0",
            "execution_count": 1,
            "outputs": [
                {
                    "output_type": "error",
                    "ename": "ZeroDivisionError",
                    "evalue": "division by zero",
                    "traceback": ["Traceback..."],
                },
            ],
            "metadata": {},
        },
    ])

    result = parse_notebook("test.ipynb", nb)

    assert result.error is None
    assert "ZeroDivisionError" in result.extracted_text
    assert "division by zero" in result.extracted_text


def test_parse_code_cell_with_image_output():
    """Test parsing a notebook with a code cell that produces an image output."""
    nb = _make_notebook([
        {
            "cell_type": "code",
            "source": "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.show()",
            "execution_count": 1,
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "image/png": "iVBORw0KGgoAAAANSUhEUgAA...",
                        "text/plain": "<Figure size 640x480 with 1 Axes>",
                    },
                    "metadata": {},
                },
            ],
            "metadata": {},
        },
    ])

    result = parse_notebook("test.ipynb", nb)

    assert result.error is None
    assert len(result.figures) == 1
    assert result.figures[0].mime_type == "image/png"
    assert result.figures[0].cell_index == 0
    assert result.figures[0].output_index == 0


def test_parse_mixed_cells():
    """Test parsing a notebook with mixed cell types."""
    nb = _make_notebook([
        {
            "cell_type": "markdown",
            "source": "# Experiment\n\n## Introduction",
            "metadata": {},
        },
        {
            "cell_type": "code",
            "source": "import numpy as np\nx = np.random.randn(100)",
            "execution_count": 1,
            "outputs": [],
            "metadata": {},
        },
        {
            "cell_type": "raw",
            "source": "This is raw text.",
            "metadata": {},
        },
    ])

    result = parse_notebook("test.ipynb", nb)

    assert result.error is None
    assert len(result.cells) == 3
    assert len(result.markdown_cells) == 1
    assert len(result.code_cells) == 1
    assert len(result.raw_cells) == 1
    assert "Experiment" in result.summary


def test_parse_invalid_json():
    """Test parsing a notebook with invalid JSON."""
    result = parse_notebook("test.ipynb", b"not valid json")

    assert result.error is not None
    assert "JSON" in result.error


def test_parse_empty_notebook():
    """Test parsing an empty notebook."""
    nb = _make_notebook([])
    result = parse_notebook("test.ipynb", nb)

    assert result.error is None
    assert len(result.cells) == 0
    assert "0 cells" in result.summary


def test_parse_summary_from_markdown():
    """Test that the summary is built from the first few markdown cells."""
    nb = _make_notebook([
        {
            "cell_type": "markdown",
            "source": "# My Experiment\n\nThis is a test.",
            "metadata": {},
        },
        {
            "cell_type": "markdown",
            "source": "## Methods\n\nWe used method X.",
            "metadata": {},
        },
    ])

    result = parse_notebook("test.ipynb", nb)

    assert result.error is None
    assert "My Experiment" in result.summary
    assert "Methods" in result.summary


def test_parse_tags():
    """Test that tags are extracted from notebook metadata."""
    nb = _make_notebook([
        {
            "cell_type": "code",
            "source": "print('hello')",
            "execution_count": 1,
            "outputs": [],
            "metadata": {"tags": ["experiment", "test"]},
        },
    ])

    result = parse_notebook("test.ipynb", nb)

    assert result.error is None
    assert "experiment" in result.tags
    assert "test" in result.tags
    assert "python" in result.tags
    assert "python3" in result.tags
