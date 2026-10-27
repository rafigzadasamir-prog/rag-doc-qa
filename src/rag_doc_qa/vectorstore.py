"""Thin wrapper around a persistent chromadb collection."""

from __future__ import annotations

from typing import Any, TypedDict

import chromadb


class RetrievedRecord(TypedDict):
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class VectorStore:
    """A single persistent chromadb collection, queried by cosine distance."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        # "hnsw:space": "cosine" makes chroma rank by cosine distance (1 - cosine similarity)
        # rather than its default squared-L2 metric, matching the task's similarity measure.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if not ids:
            return
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedRecord]:
        if self._collection.count() == 0:
            return []
        n_results = min(top_k, self._collection.count())
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        records: list[RetrievedRecord] = []
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        for i in range(len(ids)):
            records.append(
                RetrievedRecord(
                    id=ids[i],
                    text=documents[i],
                    metadata=metadatas[i],
                    distance=distances[i],
                )
            )
        return records

    def count(self) -> int:
        return self._collection.count()
