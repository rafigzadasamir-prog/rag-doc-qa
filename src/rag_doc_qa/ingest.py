"""Document ingestion: text extraction, chunking, embedding, and storage."""

from __future__ import annotations

import re
import uuid
from io import BytesIO
from typing import Any

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from .chunking import DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_CHUNK_SIZE_TOKENS, chunk_text
from .embeddings import embed_texts
from .vectorstore import VectorStore


def extract_text(filename: str, raw_bytes: bytes, content_type: str | None = None) -> str:
    """Extract plain text from a plain-text or PDF document."""
    is_pdf = (content_type == "application/pdf") or filename.lower().endswith(".pdf")
    if is_pdf:
        try:
            reader = PdfReader(BytesIO(raw_bytes))
        except Exception as exc:  # pypdf raises a variety of exception types on malformed PDFs
            raise ValueError(f"Could not parse PDF '{filename}': {exc}") from exc
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return raw_bytes.decode("utf-8", errors="replace")


def _slugify(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()
    return slug or "document"


def ingest_document(
    store: VectorStore,
    embedder: SentenceTransformer,
    filename: str,
    raw_bytes: bytes,
    content_type: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> dict[str, Any]:
    """Ingest one uploaded document into the vector store and return a summary."""
    text = extract_text(filename, raw_bytes, content_type)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError(f"No extractable text content found in '{filename}'")

    document_id = f"{_slugify(filename)}-{uuid.uuid4().hex[:8]}"
    ids = [f"{document_id}-chunk-{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": filename, "document_id": document_id, "chunk_index": i}
        for i in range(len(chunks))
    ]
    embeddings = embed_texts(chunks, embedder)
    store.add_chunks(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)

    return {
        "document_id": document_id,
        "filename": filename,
        "num_chunks": len(chunks),
    }
