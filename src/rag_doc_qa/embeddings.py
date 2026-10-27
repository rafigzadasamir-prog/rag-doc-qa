"""Local CPU embedding via sentence-transformers.

Models are cached per process by name so that ingesting several documents, or ingesting then
immediately asking a question, only pays the (relatively expensive) model-load cost once.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from .config import DEFAULT_EMBEDDING_MODEL

_EMBEDDER_CACHE: dict[str, SentenceTransformer] = {}


def load_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Load (and cache) a sentence-transformers model by name."""
    if model_name not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _EMBEDDER_CACHE[model_name]


def embed_texts(texts: list[str], embedder: SentenceTransformer) -> list[list[float]]:
    """Embed a batch of texts with an already-loaded embedder, returning plain Python lists."""
    if not texts:
        return []
    vectors = embedder.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()
