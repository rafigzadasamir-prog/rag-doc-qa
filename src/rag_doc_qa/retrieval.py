"""Query-time retrieval of relevant chunks from the vector store."""

from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from .embeddings import embed_texts
from .vectorstore import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    document_id: str
    chunk_index: int
    score: float  # cosine similarity in [-1, 1]; higher is more relevant


def retrieve_chunks(
    store: VectorStore,
    embedder: SentenceTransformer,
    question: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Embed ``question`` and return the top-k most similar chunks, best first."""
    query_embedding = embed_texts([question], embedder)[0]
    records = store.query(query_embedding, top_k)
    chunks: list[RetrievedChunk] = []
    for record in records:
        metadata = record["metadata"]
        # chroma's cosine "distance" is 1 - cosine_similarity; invert it back to a similarity
        # score so API consumers see the usual +1 (identical) to -1 (opposite) similarity scale.
        similarity = 1.0 - record["distance"]
        chunks.append(
            RetrievedChunk(
                text=record["text"],
                source=metadata.get("source", "unknown"),
                document_id=metadata.get("document_id", "unknown"),
                chunk_index=metadata.get("chunk_index", -1),
                score=similarity,
            )
        )
    return chunks
