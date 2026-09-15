"""Security checks for the native RAG service entrypoint."""

from pathlib import Path


def test_rag_service_defaults_to_loopback_bind():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src/umrs/api/rag_service.py").read_text()

    assert 'os.getenv("UMRS_RAG_BIND_HOST", "127.0.0.1")' in source
    assert 'host="0.0.0.0"' not in source
