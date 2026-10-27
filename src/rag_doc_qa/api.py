"""FastAPI application: document upload and RAG question answering."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import Settings, get_settings
from .embeddings import load_embedder
from .generation import generate_answer
from .ingest import ingest_document
from .retrieval import retrieve_chunks
from .schemas import AskRequest, AskResponse, HealthResponse, IngestResponse, SourceRef
from .vectorstore import VectorStore


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. Accepts explicit ``settings`` so tests can inject an isolated
    chroma persistence directory and a controlled ANTHROPIC_API_KEY without relying on global
    environment state.
    """
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved_settings
        app.state.embedder = load_embedder(resolved_settings.embedding_model_name)
        app.state.store = VectorStore(
            resolved_settings.chroma_persist_dir, resolved_settings.collection_name
        )
        yield

    app = FastAPI(
        title="RAG Document Q&A",
        description=(
            "Upload documents, then ask grounded questions over them. Falls back to a "
            "deterministic extractive answer when no ANTHROPIC_API_KEY is configured."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", documents_indexed=app.state.store.count())

    @app.post("/documents", response_model=IngestResponse)
    async def upload_document(file: Annotated[UploadFile, File()]) -> IngestResponse:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        try:
            result = ingest_document(
                store=app.state.store,
                embedder=app.state.embedder,
                filename=file.filename or "untitled.txt",
                raw_bytes=raw_bytes,
                content_type=file.content_type,
                chunk_size=app.state.settings.chunk_size_tokens,
                overlap=app.state.settings.chunk_overlap_tokens,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IngestResponse(**result)

    @app.post("/ask", response_model=AskResponse)
    async def ask_question(payload: AskRequest) -> AskResponse:
        settings = app.state.settings
        top_k = payload.top_k or settings.default_top_k
        chunks = retrieve_chunks(app.state.store, app.state.embedder, payload.question, top_k)
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No documents have been ingested yet. Upload one via POST /documents.",
            )
        result = generate_answer(payload.question, chunks, settings)
        return AskResponse(
            question=payload.question,
            answer=result["answer"],
            mode=result["mode"],
            sources=[SourceRef(**source) for source in result["sources"]],
        )

    return app


app = create_app()
