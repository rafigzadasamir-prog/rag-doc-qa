"""Pydantic request/response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer.")
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Number of chunks to retrieve."
    )


class SourceRef(BaseModel):
    source: str
    chunk_index: int
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: str
    sources: list[SourceRef]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
