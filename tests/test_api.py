from pathlib import Path

from fastapi.testclient import TestClient

from rag_doc_qa.api import create_app
from rag_doc_qa.config import Settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _offline_settings(tmp_path: Path) -> Settings:
    return Settings(
        chroma_persist_dir=str(tmp_path / "chroma"),
        collection_name="test-documents",
        embedding_model_name="all-MiniLM-L6-v2",
        chunk_size_tokens=500,
        chunk_overlap_tokens=50,
        default_top_k=3,
        anthropic_model="claude-sonnet-5",
        anthropic_api_key=None,
    )


def test_health_reports_zero_documents_on_a_fresh_store(tmp_path):
    app = create_app(_offline_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "documents_indexed": 0}


def test_ask_before_any_upload_returns_404(tmp_path):
    app = create_app(_offline_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "anything?"})
        assert response.status_code == 404


def test_upload_rejects_empty_file(tmp_path):
    app = create_app(_offline_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post("/documents", files={"file": ("empty.txt", b"", "text/plain")})
        assert response.status_code == 400


def test_end_to_end_ingest_then_ask_offline(tmp_path):
    """Full offline RAG loop: upload the bundled sample document, ask a grounded question with
    no ANTHROPIC_API_KEY configured, and confirm the extractive answer is the correct chunk."""
    app = create_app(_offline_settings(tmp_path))
    with TestClient(app) as client:
        with open(DATA_DIR / "sample_document.txt", "rb") as f:
            upload_response = client.post(
                "/documents", files={"file": ("sample_document.txt", f, "text/plain")}
            )
        assert upload_response.status_code == 200
        upload_body = upload_response.json()
        assert upload_body["filename"] == "sample_document.txt"
        assert upload_body["num_chunks"] > 1

        ask_response = client.post(
            "/ask",
            json={
                "question": "What encryption algorithm does Nimbus use to protect data at rest?",
                "top_k": 2,
            },
        )
        assert ask_response.status_code == 200
        payload = ask_response.json()
        assert payload["mode"] == "extractive"
        assert "aes-256 encryption" in payload["answer"].lower()
        assert payload["sources"][0]["source"] == "sample_document.txt"
        assert len(payload["sources"]) == 2


def test_ask_validates_question_is_non_empty(tmp_path):
    app = create_app(_offline_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422
