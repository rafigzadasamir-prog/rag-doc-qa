"""Answer generation: real Anthropic call when an API key is configured, otherwise a
deterministic extractive fallback so the service always works offline and in CI with zero
network calls.
"""

from __future__ import annotations

from typing import Any

import anthropic

from .config import Settings
from .retrieval import RetrievedChunk

_SYSTEM_PROMPT = (
    "You are a document question-answering assistant. Answer the user's question using ONLY "
    "the provided context excerpts. If the answer is not contained in the context, say you "
    "don't know rather than guessing. Be concise, and mention which source file supports your "
    "answer."
)


def _build_client() -> anthropic.Anthropic:
    """Isolated so tests can monkeypatch client construction without touching the network."""
    return anthropic.Anthropic()


def _sources_payload(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "score": round(chunk.score, 4),
        }
        for chunk in chunks
    ]


def _generate_extractive(question: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
    """Return the single most relevant chunk verbatim, plus its source - no model call."""
    best = chunks[0]
    return {
        "answer": best.text,
        "mode": "extractive",
        "sources": _sources_payload(chunks),
    }


def _generate_with_anthropic(
    question: str, chunks: list[RetrievedChunk], settings: Settings
) -> dict[str, Any]:
    context = "\n\n".join(
        f"[Source: {chunk.source}, chunk {chunk.chunk_index}]\n{chunk.text}" for chunk in chunks
    )
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    client = _build_client()
    try:
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.APIStatusError:
        # The context has already been retrieved; degrade to the offline answer rather than
        # failing the whole request just because the upstream API had a bad moment.
        result = _generate_extractive(question, chunks)
        result["mode"] = "extractive_fallback_api_error"
        return result

    text = "".join(block.text for block in response.content if block.type == "text")
    return {
        "answer": text,
        "mode": "anthropic",
        "sources": _sources_payload(chunks),
    }


def generate_answer(
    question: str, chunks: list[RetrievedChunk], settings: Settings
) -> dict[str, Any]:
    """Generate an answer for ``question`` grounded in the retrieved ``chunks``.

    Uses the real Anthropic API when ``settings.anthropic_api_key`` is set, otherwise falls
    back to a deterministic extractive answer with no network access.
    """
    if not chunks:
        return {"answer": "", "mode": "no_context", "sources": []}
    if settings.anthropic_api_key:
        return _generate_with_anthropic(question, chunks, settings)
    return _generate_extractive(question, chunks)
