from pathlib import Path

import pytest

from rag_doc_qa.embeddings import load_embedder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(autouse=True)
def _default_offline_mode(monkeypatch):
    """Every test starts with no Anthropic credentials configured, forcing the deterministic
    extractive path. Tests that want to exercise the Anthropic branch set the env var (and mock
    the client) explicitly within the test itself.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(scope="session")
def shared_embedder():
    """Loading the sentence-transformers model is the slowest part of the suite - share one
    instance across every test instead of reloading it per test.
    """
    return load_embedder()


@pytest.fixture
def sample_document_bytes() -> bytes:
    return (DATA_DIR / "sample_document.txt").read_bytes()
