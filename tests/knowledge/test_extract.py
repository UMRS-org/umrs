"""Tests for the knowledge extraction module."""

from __future__ import annotations

import json

import pytest

from umrs.knowledge.extract import (
    ExtractedKnowledge,
    KnowledgeExtractor,
    extract_knowledge,
)


# ── Test Fixtures ────────────────────────────────────────────────────────────


PAPER_TEXT = """
Machine Learning for Scientific Discovery

Abstract
In this paper, we present a novel approach to machine learning for scientific
discovery. The goal of this paper is to demonstrate that transformer-based
models can be used to predict molecular properties with high accuracy.

1. Introduction
We aim to develop a framework that combines deep learning with quantum
chemistry calculations. In this work, we introduce a new architecture that
bridges the gap between traditional ML and quantum mechanical methods.

2. Methodology
Our method consists of three components: (1) a graph neural network for
molecular representation, (2) a transformer encoder for property prediction,
and (3) a fine-tuning strategy that leverages pre-trained language models.
We use the QM9 dataset for training and evaluation.

3. Experiments
We evaluate our approach on three benchmark datasets: QM9, ZINC, and
MD17. Our experimental setup includes 5-fold cross-validation with a
80/20 train/test split. We use learning_rate=0.001 and batch_size=32.

4. Results
Our results show that our approach achieves state-of-the-art performance
on all three benchmarks. We conclude that transformer-based models can
effectively predict molecular properties when combined with graph neural
networks.

5. Limitations
However, a limitation is that our approach requires large amounts of
training data. One limitation is that the model struggles with rare
molecular structures.

Keywords: machine learning, molecular properties, transformers, graph neural networks
"""


@pytest.fixture
def paper_content():
    """Create paper content bytes."""
    return PAPER_TEXT.encode("utf-8")


@pytest.fixture
def code_content():
    """Create Python code content bytes."""
    code = b'''
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """A transformer model for molecular property prediction."""

    def __init__(self, input_dim=128, hidden_size=256, num_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8, dropout=dropout),
            num_layers=num_layers
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)

# Experimental settings
learning_rate = 0.001
batch_size = 32
epochs = 100
seed = 42

def main():
    model = TransformerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop...

if __name__ == "__main__":
    main()
'''
    return code


@pytest.fixture
def notebook_content():
    """Create Jupyter notebook content bytes."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "language": "python",
            },
        },
        "cells": [
            {
                "cell_type": "markdown",
                "source": "# Molecular Property Prediction\n\nIn this notebook, we demonstrate how to use transformer models for molecular property prediction.",
                "metadata": {},
            },
            {
                "cell_type": "code",
                "source": "import torch\nimport torch.nn as nn\n\n# Experimental settings\nlearning_rate = 0.001\nbatch_size = 32",
                "execution_count": 1,
                "outputs": [],
                "metadata": {},
            },
            {
                "cell_type": "code",
                "source": "model = TransformerModel()\nprint(model)",
                "execution_count": 2,
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": "TransformerModel(\n  (embedding): Linear(in_features=128, out_features=256)\n)\n",
                    },
                ],
                "metadata": {},
            },
            {
                "cell_type": "code",
                "source": "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.show()",
                "execution_count": 3,
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {
                            "image/png": "iVBORw0KGgoAAAANSUhEUgAA...",
                            "text/plain": "<Figure>",
                        },
                        "metadata": {},
                    },
                ],
                "metadata": {},
            },
        ],
    }
    return json.dumps(nb).encode("utf-8")


# ── ExtractedKnowledge Tests ───────────────────────────────────────────────


def test_extracted_knowledge_defaults():
    """Test ExtractedKnowledge defaults."""
    knowledge = ExtractedKnowledge(
        artifact_id="test-paper",
        artifact_type="paper",
        filename="paper.pdf",
    )

    assert knowledge.artifact_id == "test-paper"
    assert knowledge.artifact_type == "paper"
    assert knowledge.filename == "paper.pdf"
    assert knowledge.objective == ""
    assert knowledge.assumptions == []
    assert knowledge.extraction_confidence == 0.0


def test_extracted_knowledge_to_dict():
    """Test ExtractedKnowledge to_dict serialization."""
    knowledge = ExtractedKnowledge(
        artifact_id="test-paper",
        artifact_type="paper",
        filename="paper.pdf",
        objective="Test objective",
        methodology="Test methodology",
        conclusions=["Conclusion 1", "Conclusion 2"],
    )

    data = knowledge.to_dict()

    assert data["artifact_id"] == "test-paper"
    assert data["objective"] == "Test objective"
    assert data["conclusions"] == ["Conclusion 1", "Conclusion 2"]
    assert "methodology" in data
    assert "extraction_confidence" in data


# ── KnowledgeExtractor Tests ───────────────────────────────────────────────


def test_extractor_init():
    """Test KnowledgeExtractor initialization."""
    extractor = KnowledgeExtractor()

    assert len(extractor._objective_patterns) > 0
    assert len(extractor._methodology_patterns) > 0
    assert len(extractor._conclusion_patterns) > 0


def test_extract_from_paper(paper_content):
    """Test extracting knowledge from a paper."""
    extractor = KnowledgeExtractor()
    knowledge = extractor.extract(
        artifact_id="paper-001",
        artifact_type="paper",
        filename="paper.pdf",
        content=paper_content,
    )

    assert knowledge.artifact_id == "paper-001"
    assert knowledge.artifact_type == "paper"
    # Should extract some objective
    assert knowledge.objective != "" or knowledge.summary != ""
    # Should extract some keywords
    assert len(knowledge.keywords) > 0
    assert "machine learning" in knowledge.keywords
    # Should have a summary
    assert knowledge.summary != ""
    # Should have tags
    assert "paper" in knowledge.tags


def test_extract_from_code(code_content):
    """Test extracting knowledge from code."""
    extractor = KnowledgeExtractor()
    knowledge = extractor.extract(
        artifact_id="code-001",
        artifact_type="code",
        filename="model.py",
        content=code_content,
    )

    assert knowledge.artifact_id == "code-001"
    assert knowledge.artifact_type == "code"
    # Should extract modules
    assert len(knowledge.modules) > 0
    # Should extract APIs
    assert len(knowledge.apis) > 0
    # Should extract dependencies
    assert "torch" in knowledge.dependencies
    # Should extract parameters
    assert "input_dim" in knowledge.parameters or "hidden_size" in knowledge.parameters
    # Should extract execution flow
    assert knowledge.execution_flow == "has_main_block"
    # Should extract experimental settings
    assert len(knowledge.experimental_settings) > 0
    assert any("learning_rate" in s for s in knowledge.experimental_settings)


def test_extract_from_notebook(notebook_content):
    """Test extracting knowledge from a notebook."""
    extractor = KnowledgeExtractor()
    knowledge = extractor.extract(
        artifact_id="notebook-001",
        artifact_type="notebook",
        filename="experiment.ipynb",
        content=notebook_content,
    )

    assert knowledge.artifact_id == "notebook-001"
    assert knowledge.artifact_type == "notebook"
    # Should extract markdown explanations
    assert len(knowledge.markdown_explanations) > 0
    assert "Molecular Property Prediction" in knowledge.markdown_explanations[0]
    # Should extract figures
    assert len(knowledge.figures) == 1
    assert knowledge.figures[0]["mime_type"] == "image/png"
    # Should extract generated outputs
    assert len(knowledge.generated_outputs) > 0
    assert "TransformerModel" in knowledge.generated_outputs[0]
    # Should extract execution order
    assert knowledge.execution_order == [1, 2, 3]
    # Should extract experimental settings
    assert len(knowledge.experimental_settings) > 0
    assert any("learning_rate" in s for s in knowledge.experimental_settings)


def test_extract_from_text():
    """Test extracting knowledge from text."""
    text_content = b"""
This is a test document.

The goal of this paper is to test the extraction pipeline.
We propose a new method for knowledge extraction.
We conclude that the extraction works correctly.

Keywords: test, extraction, pipeline
"""

    extractor = KnowledgeExtractor()
    knowledge = extractor.extract(
        artifact_id="text-001",
        artifact_type="text",
        filename="document.txt",
        content=text_content,
    )

    assert knowledge.artifact_type == "text"
    assert knowledge.objective != ""
    assert "test the extraction pipeline" in knowledge.objective
    assert len(knowledge.conclusions) > 0
    assert len(knowledge.keywords) > 0
    assert "test" in knowledge.keywords


def test_extract_from_unknown_type():
    """Test extracting knowledge from unknown artifact type."""
    extractor = KnowledgeExtractor()
    knowledge = extractor.extract(
        artifact_id="unknown-001",
        artifact_type="unknown",
        filename="unknown.xyz",
        content=b"some content",
    )

    # Should still return an ExtractedKnowledge object
    assert knowledge.artifact_type == "unknown"
    # Should not crash


# ── Quality Metrics Tests ──────────────────────────────────────────────────


def test_extraction_confidence(paper_content):
    """Test that extraction confidence is calculated."""
    extractor = KnowledgeExtractor()
    knowledge = extractor.extract(
        artifact_id="paper-001",
        artifact_type="paper",
        filename="paper.pdf",
        content=paper_content,
    )

    assert 0.0 <= knowledge.extraction_confidence <= 1.0
    # With a well-structured paper, confidence should be > 0
    assert knowledge.extraction_confidence > 0.0


def test_coverage_score(paper_content):
    """Test that coverage score is calculated."""
    extractor = KnowledgeExtractor()
    knowledge = extractor.extract(
        artifact_id="paper-001",
        artifact_type="paper",
        filename="paper.pdf",
        content=paper_content,
    )

    assert 0.0 <= knowledge.coverage_score <= 1.0
    # With a well-structured paper, coverage should be > 0
    assert knowledge.coverage_score > 0.0


# ── Convenience Function Tests ─────────────────────────────────────────────


def test_extract_knowledge_convenience_function(code_content):
    """Test the extract_knowledge convenience function."""
    knowledge = extract_knowledge(
        artifact_id="code-001",
        artifact_type="code",
        filename="model.py",
        content=code_content,
    )

    assert knowledge.artifact_id == "code-001"
    assert knowledge.artifact_type == "code"
    assert len(knowledge.modules) > 0


# ── Edge Case Tests ────────────────────────────────────────────────────────


def test_extract_empty_content():
    """Test extracting knowledge from empty content."""
    extractor = KnowledgeExtractor()
    knowledge = extractor.extract(
        artifact_id="empty-001",
        artifact_type="text",
        filename="empty.txt",
        content=b"",
    )

    assert knowledge.artifact_id == "empty-001"
    # Should not crash, should return empty fields
    assert knowledge.objective == ""


def test_extract_invalid_notebook():
    """Test extracting knowledge from invalid notebook JSON."""
    extractor = KnowledgeExtractor()
    knowledge = extractor.extract(
        artifact_id="invalid-001",
        artifact_type="notebook",
        filename="invalid.ipynb",
        content=b"not valid json",
    )

    assert knowledge.artifact_type == "notebook"
    # Should not crash, should return empty fields
    assert len(knowledge.markdown_explanations) == 0


def test_extract_paper_with_keywords():
    """Test that keywords are extracted from paper text."""
    paper_with_keywords = b"""
This paper presents a novel approach.

Keywords: machine learning, deep learning, neural networks, transformers
"""

    extractor = KnowledgeExtractor()
    knowledge = extractor.extract(
        artifact_id="paper-002",
        artifact_type="paper",
        filename="paper.pdf",
        content=paper_with_keywords,
    )

    assert len(knowledge.keywords) >= 4
    assert "machine learning" in knowledge.keywords
    assert "neural networks" in knowledge.keywords


def test_extract_code_with_algorithms():
    """Test that algorithms are extracted from code."""
    code_with_algorithm = b'''
def quicksort_algorithm(arr):
    """Sort using the quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_algorithm(left) + middle + quicksort_algorithm(right)
'''
    extractor = KnowledgeExtractor()
    knowledge = extractor.extract(
        artifact_id="code-002",
        artifact_type="code",
        filename="sort.py",
        content=code_with_algorithm,
    )

    assert len(knowledge.algorithms) > 0
    assert "quicksort_algorithm" in knowledge.algorithms
