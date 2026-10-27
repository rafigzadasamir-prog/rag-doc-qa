import pytest

from rag_doc_qa.ingest import ingest_document
from rag_doc_qa.retrieval import retrieve_chunks
from rag_doc_qa.vectorstore import VectorStore


@pytest.fixture(scope="module")
def ingested_store(tmp_path_factory, shared_embedder, sample_document_bytes):
    persist_dir = tmp_path_factory.mktemp("chroma-retrieval")
    store = VectorStore(str(persist_dir), "retrieval-test")
    ingest_document(store, shared_embedder, "sample_document.txt", sample_document_bytes)
    return store


def test_ingested_store_has_multiple_chunks(ingested_store):
    assert ingested_store.count() > 1


@pytest.mark.parametrize(
    "question,expected_substring",
    [
        (
            "What encryption algorithm does Nimbus use to protect data at rest?",
            "AES-256 encryption",
        ),
        (
            "What certification does Nimbus hold for its information security management "
            "system?",
            "ISO 27001 certification",
        ),
        (
            "What is the guaranteed first response time for critical severity incidents?",
            "30 minutes for critical severity incidents",
        ),
        ("What trial do new signups get automatically?", "14-day trial of the Professional plan"),
    ],
)
def test_top_result_contains_the_grounded_answer(
    ingested_store, shared_embedder, question, expected_substring
):
    results = retrieve_chunks(ingested_store, shared_embedder, question, top_k=3)
    assert results
    assert expected_substring.lower() in results[0].text.lower()
    assert results[0].source == "sample_document.txt"


def test_results_are_ranked_by_descending_similarity(ingested_store, shared_embedder):
    results = retrieve_chunks(
        ingested_store, shared_embedder, "How do I set up single sign-on for my team?", top_k=5
    )
    scores = [chunk.score for chunk in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_number_of_results(ingested_store, shared_embedder):
    results = retrieve_chunks(ingested_store, shared_embedder, "pricing", top_k=2)
    assert len(results) <= 2


def test_empty_store_returns_no_results(tmp_path, shared_embedder):
    empty_store = VectorStore(str(tmp_path / "empty-chroma"), "empty-retrieval-test")
    results = retrieve_chunks(empty_store, shared_embedder, "anything?", top_k=3)
    assert results == []
