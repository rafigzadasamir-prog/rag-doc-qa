from types import SimpleNamespace

from rag_doc_qa.config import get_settings
from rag_doc_qa.generation import generate_answer
from rag_doc_qa.retrieval import RetrievedChunk


def _sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text="AES-256 encryption is used to protect data at rest.",
            source="doc.txt",
            document_id="doc",
            chunk_index=0,
            score=0.91,
        ),
        RetrievedChunk(
            text="TLS 1.3 protects data in transit.",
            source="doc.txt",
            document_id="doc",
            chunk_index=1,
            score=0.47,
        ),
    ]


def test_extractive_mode_returns_top_chunk_verbatim(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = get_settings()

    chunks = _sample_chunks()
    result = generate_answer("What encrypts data at rest?", chunks, settings)

    assert result["mode"] == "extractive"
    assert result["answer"] == chunks[0].text
    assert result["sources"][0]["source"] == "doc.txt"
    assert result["sources"][0]["chunk_index"] == 0


def test_no_retrieved_context_short_circuits(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = get_settings()

    result = generate_answer("anything", [], settings)

    assert result == {"answer": "", "mode": "no_context", "sources": []}


def test_anthropic_mode_uses_mocked_client_no_network_call(monkeypatch):
    """Verifies the Anthropic code path end to end while guaranteeing zero network access:
    the client constructor itself is replaced with a fake that records the call and returns a
    canned message object shaped like the real SDK's response.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-a-real-credential")
    settings = get_settings()
    assert settings.anthropic_api_key == "test-key-not-a-real-credential"

    captured_kwargs = {}

    class _FakeMessages:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            block = SimpleNamespace(type="text", text="Mocked grounded answer about encryption.")
            return SimpleNamespace(content=[block])

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("rag_doc_qa.generation._build_client", lambda: _FakeClient())

    result = generate_answer("What encrypts data at rest?", _sample_chunks(), settings)

    assert result["mode"] == "anthropic"
    assert result["answer"] == "Mocked grounded answer about encryption."
    assert captured_kwargs["model"] == settings.anthropic_model
    assert "AES-256" in captured_kwargs["messages"][0]["content"]


def test_anthropic_api_error_falls_back_to_extractive(monkeypatch):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-a-real-credential")
    settings = get_settings()

    class _FakeMessages:
        def create(self, **kwargs):
            request = SimpleNamespace()
            response = SimpleNamespace(status_code=500, headers={}, request=request)
            raise anthropic.APIStatusError(
                "simulated upstream failure", response=response, body=None
            )

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("rag_doc_qa.generation._build_client", lambda: _FakeClient())

    chunks = _sample_chunks()
    result = generate_answer("What encrypts data at rest?", chunks, settings)

    assert result["mode"] == "extractive_fallback_api_error"
    assert result["answer"] == chunks[0].text
