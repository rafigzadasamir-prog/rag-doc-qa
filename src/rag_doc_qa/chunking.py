"""Sliding-window text chunker.

Token counts are approximated by whitespace-split word counts rather than a model-specific
tokenizer. This keeps chunking dependency-free and fully deterministic (no tokenizer download,
no network call), which matters for the offline/CI story of this service - the approximation is
close enough for ~500-token chunks that the difference against a real BPE tokenizer does not
change retrieval quality in practice.
"""

from __future__ import annotations

DEFAULT_CHUNK_SIZE_TOKENS = 500

# A 10% overlap is enough to keep a sentence that straddles a chunk boundary readable in both
# of its neighboring chunks, without the duplicate-content tax that a larger overlap (and the
# extra embedding/storage cost that comes with it) would impose on every document.
DEFAULT_CHUNK_OVERLAP_TOKENS = 50


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split ``text`` into overlapping, whitespace-token-bounded chunks.

    Returns an empty list for blank/whitespace-only input. Raises ``ValueError`` for a
    non-positive chunk size or an overlap that is not strictly smaller than the chunk size
    (otherwise the sliding window never advances).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = text.split()
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while True:
        window = tokens[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break
        start += step
    return chunks
