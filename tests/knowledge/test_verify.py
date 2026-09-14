"""Tests for the knowledge verification chat (Phase 6)."""

from __future__ import annotations

import pytest
from umrs.knowledge.verify import (
    Evidence,
    EvidenceType,
    InMemoryKnowledgeStore,
    KnowledgeVerifier,
    VerificationAnswer,
    verify_knowledge,
)

# ── Test Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def empty_store():
    """Create an empty knowledge store."""
    return InMemoryKnowledgeStore()


@pytest.fixture
def populated_store():
    """Create a knowledge store with test artifacts."""
    store = InMemoryKnowledgeStore()

    # Add a paper
    store.add_artifact(
        {
            "artifact_id": "paper-001",
            "artifact_type": "paper",
            "filename": "attention_is_all_you_need.pdf",
            "title": "Attention Is All You Need",
            "tags": ["transformer", "attention"],
        }
    )
    store.add_extracted_knowledge(
        "paper-001",
        {
            "objective": "Present transformer architecture",
            "methodology": "Self-attention mechanism",
            "keywords": ["transformer", "attention", "neural networks"],
            "algorithms": ["transformer", "self-attention"],
            "conclusions": ["Transformers outperform RNNs"],
            "assumptions": ["Self-attention is sufficient"],
        },
    )

    # Add code
    store.add_artifact(
        {
            "artifact_id": "code-001",
            "artifact_type": "code",
            "filename": "transformer.py",
            "tags": ["python", "transformer"],
        }
    )
    store.add_extracted_knowledge(
        "code-001",
        {
            "algorithms": ["transformer", "self-attention"],
            "apis": ["TransformerModel", "MultiHeadAttention"],
            "modules": ["transformer"],
            "dependencies": ["torch", "numpy"],
            "parameters": ["d_model", "nhead", "num_layers"],
            "kernels": [],
            "execution_flow": "has_main_block",
        },
    )

    # Add a notebook
    store.add_artifact(
        {
            "artifact_id": "notebook-001",
            "artifact_type": "notebook",
            "filename": "transformer_experiment.ipynb",
            "tags": ["experiment", "transformer"],
        }
    )
    store.add_extracted_knowledge(
        "notebook-001",
        {
            "algorithms": ["transformer"],
            "figures": [
                {"cell_index": 5, "output_index": 0, "mime_type": "image/png"},
                {"cell_index": 8, "output_index": 0, "mime_type": "image/png"},
            ],
            "generated_outputs": ["Training loss: 0.5", "Validation accuracy: 0.85"],
            "execution_order": [1, 2, 3, 4, 5],
            "experimental_settings": ["learning_rate=0.001", "batch_size=32"],
        },
    )

    # Add a dataset
    store.add_artifact(
        {
            "artifact_id": "dataset-001",
            "artifact_type": "dataset",
            "filename": "qm9.csv",
            "tags": ["molecules", "qm9"],
        }
    )

    return store


# ── Evidence Model Tests ──────────────────────────────────────────────────


def test_evidence_defaults():
    """Test Evidence defaults."""
    evidence = Evidence(evidence_type=EvidenceType.UPLOADED_FILE)

    assert evidence.evidence_type == EvidenceType.UPLOADED_FILE
    assert evidence.source_id == ""
    assert evidence.confidence == 0.0


def test_evidence_to_dict():
    """Test Evidence to_dict."""
    evidence = Evidence(
        evidence_type=EvidenceType.NOTEBOOK_CELL,
        source_id="nb-001",
        source_name="experiment.ipynb",
        content="Cell output",
        confidence=0.9,
    )

    data = evidence.to_dict()
    assert data["evidence_type"] == "notebook_cell"
    assert data["source_id"] == "nb-001"
    assert data["confidence"] == 0.9


# ── VerificationAnswer Tests ──────────────────────────────────────────────


def test_verification_answer_defaults():
    """Test VerificationAnswer defaults."""
    answer = VerificationAnswer(question="test?", answer="test answer")

    assert answer.question == "test?"
    assert answer.answer == "test answer"
    assert answer.evidence == []
    assert answer.confidence == 0.0
    assert answer.missing_info == []
    assert answer.answered is True


def test_verification_answer_to_dict():
    """Test VerificationAnswer to_dict."""
    answer = VerificationAnswer(
        question="What algorithms?",
        answer="Found 2 algorithms",
        evidence=[Evidence(evidence_type=EvidenceType.SOURCE_CODE)],
        confidence=0.8,
        missing_info=["No paper support"],
    )

    data = answer.to_dict()
    assert data["question"] == "What algorithms?"
    assert data["confidence"] == 0.8
    assert len(data["evidence"]) == 1
    assert data["missing_info"] == ["No paper support"]


# ── InMemoryKnowledgeStore Tests ──────────────────────────────────────────


def test_in_memory_store_add_artifact():
    """Test adding artifacts to the store."""
    store = InMemoryKnowledgeStore()
    store.add_artifact(
        {
            "artifact_id": "test-001",
            "artifact_type": "paper",
            "filename": "test.pdf",
        }
    )

    assert len(store.list_artifacts()) == 1
    assert store.get_artifact("test-001") is not None


def test_in_memory_store_list_by_type():
    """Test listing artifacts by type."""
    store = InMemoryKnowledgeStore()
    store.add_artifact(
        {
            "artifact_id": "p1",
            "artifact_type": "paper",
            "filename": "p1.pdf",
        }
    )
    store.add_artifact(
        {
            "artifact_id": "c1",
            "artifact_type": "code",
            "filename": "c1.py",
        }
    )

    assert len(store.list_artifacts("paper")) == 1
    assert len(store.list_artifacts("code")) == 1
    assert len(store.list_artifacts()) == 2


def test_in_memory_store_search():
    """Test searching artifacts."""
    store = InMemoryKnowledgeStore()
    store.add_artifact(
        {
            "artifact_id": "t1",
            "artifact_type": "paper",
            "filename": "transformer.pdf",
            "tags": ["transformer"],
        }
    )
    store.add_artifact(
        {
            "artifact_id": "t2",
            "artifact_type": "code",
            "filename": "cnn.py",
            "tags": ["cnn"],
        }
    )

    results = store.search_artifacts("transformer")
    assert len(results) == 1
    assert results[0]["artifact_id"] == "t1"


# ── KnowledgeVerifier Tests ───────────────────────────────────────────────


def test_verifier_explain_project_empty(empty_store):
    """Test explaining an empty project."""
    verifier = KnowledgeVerifier(empty_store)
    answer = verifier.verify("Explain my project")

    assert answer.question == "Explain my project"
    assert "No artifacts" in answer.answer or "0 artifacts" in answer.answer
    assert answer.confidence == 0.0


def test_verifier_explain_project_populated(populated_store):
    """Test explaining a populated project."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("Explain my project")

    assert "artifacts" in answer.answer
    assert answer.confidence > 0.0


def test_verifier_list_algorithms(populated_store):
    """Test listing algorithms."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("What algorithms are implemented?")

    assert "transformer" in answer.answer.lower()
    assert answer.confidence > 0.0
    assert len(answer.evidence) > 0


def test_verifier_list_papers(populated_store):
    """Test listing papers."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("Which papers do we have?")

    assert "Attention" in answer.answer or "paper" in answer.answer.lower()
    assert answer.confidence > 0.0


def test_verifier_list_code(populated_store):
    """Test listing code files."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("Which code files do we have?")

    assert "transformer.py" in answer.answer
    assert answer.confidence > 0.0


def test_verifier_list_notebooks(populated_store):
    """Test listing notebooks."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("Which notebooks do we have?")

    assert "transformer_experiment.ipynb" in answer.answer
    assert answer.confidence > 0.0


def test_verifier_list_datasets(populated_store):
    """Test listing datasets."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("Which datasets do we have?")

    assert "qm9.csv" in answer.answer
    assert answer.confidence > 0.0


def test_verifier_list_assumptions(populated_store):
    """Test listing assumptions."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("Which assumptions remain unverified?")

    assert "Self-attention" in answer.answer or "assumption" in answer.answer.lower()
    assert answer.confidence > 0.0


def test_verifier_list_conclusions(populated_store):
    """Test listing conclusions."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("Which conclusions do we have?")

    assert "Transformers" in answer.answer or "conclusion" in answer.answer.lower()
    assert answer.confidence > 0.0


def test_verifier_how_figure_generated(populated_store):
    """Test asking how a figure was generated."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("How was Figure 5 generated?")

    assert "figures" in answer.answer.lower() or "notebook" in answer.answer.lower()


def test_verifier_which_notebook_produced(populated_store):
    """Test asking which notebook produced a dataset."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("Which notebook produced this dataset?")

    assert "notebook" in answer.answer.lower() or "outputs" in answer.answer.lower()


def test_verifier_which_papers_support(populated_store):
    """Test asking which papers support an implementation."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("Which papers support this implementation?")

    assert "paper" in answer.answer.lower() or "support" in answer.answer.lower()


def test_verifier_generic_question(populated_store):
    """Test a generic/unrecognized question."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("What is the meaning of life?")

    assert answer.question == "What is the meaning of life?"
    # Should return some answer
    assert isinstance(answer.answer, str)


def test_verifier_unknown_question_empty_store(empty_store):
    """Test unknown question with empty store."""
    verifier = KnowledgeVerifier(empty_store)
    answer = verifier.verify("random question")

    assert answer.confidence == 0.0
    assert "No matching artifacts" in answer.missing_info or len(answer.missing_info) > 0


# ── Convenience Function Tests ─────────────────────────────────────────────


def test_verify_knowledge_convenience_function(populated_store):
    """Test the verify_knowledge convenience function."""
    answer = verify_knowledge("What algorithms are implemented?", populated_store)

    assert isinstance(answer, VerificationAnswer)
    assert "transformer" in answer.answer.lower()


# ── Edge Case Tests ────────────────────────────────────────────────────────


def test_verifier_empty_question(populated_store):
    """Test an empty question."""
    verifier = KnowledgeVerifier(populated_store)
    answer = verifier.verify("")

    assert answer.question == ""
    assert isinstance(answer.answer, str)


def test_verifier_case_insensitive(populated_store):
    """Test that question matching is case-insensitive."""
    verifier = KnowledgeVerifier(populated_store)

    answer_lower = verifier.verify("what algorithms are implemented?")
    answer_upper = verifier.verify("WHAT ALGORITHMS ARE IMPLEMENTED?")

    assert answer_lower.confidence == answer_upper.confidence


def test_evidence_type_enum():
    """Test EvidenceType enum values."""
    assert EvidenceType.UPLOADED_FILE.value == "uploaded_file"
    assert EvidenceType.NOTEBOOK_CELL.value == "notebook_cell"
    assert EvidenceType.SOURCE_CODE.value == "source_code"
    assert EvidenceType.EXTERNAL_PAPER.value == "external_paper"
    assert EvidenceType.WEB_SEARCH.value == "web_search"
