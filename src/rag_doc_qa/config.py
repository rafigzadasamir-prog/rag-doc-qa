"""Runtime configuration, read fresh from the environment on every call.

``get_settings`` deliberately re-reads ``os.environ`` on each invocation (rather than caching
a module-level singleton) so tests can monkeypatch environment variables - e.g. an isolated
``CHROMA_PERSIST_DIR`` per test, or removing ``ANTHROPIC_API_KEY`` to force the offline path -
and immediately see the effect via a fresh ``create_app()`` / ``get_settings()`` call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"


@dataclass(frozen=True)
class Settings:
    chroma_persist_dir: str
    collection_name: str
    embedding_model_name: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    default_top_k: int
    anthropic_model: str
    anthropic_api_key: str | None


def get_settings() -> Settings:
    return Settings(
        chroma_persist_dir=os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data"),
        collection_name=os.environ.get("CHROMA_COLLECTION_NAME", "documents"),
        embedding_model_name=os.environ.get("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL),
        chunk_size_tokens=int(os.environ.get("CHUNK_SIZE_TOKENS", "500")),
        chunk_overlap_tokens=int(os.environ.get("CHUNK_OVERLAP_TOKENS", "50")),
        default_top_k=int(os.environ.get("DEFAULT_TOP_K", "3")),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
    )
