#!/usr/bin/env python
"""Evaluate retrieval quality (precision@k / recall@k) against a tiny bundled Q/A dataset.

Ground truth is derived dynamically: a chunk counts as "relevant" to a question if it contains
that question's expected answer substring (see data/eval_dataset.json). Deriving relevance
after chunking - rather than hardcoding chunk indices - keeps the eval valid no matter how the
chunk-size/overlap settings change the exact chunk boundaries.

Usage:
    python scripts/evaluate_retrieval.py [--top-k 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_doc_qa.chunking import chunk_text  # noqa: E402
from rag_doc_qa.config import get_settings  # noqa: E402
from rag_doc_qa.embeddings import embed_texts, load_embedder  # noqa: E402
from rag_doc_qa.vectorstore import VectorStore  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_eval_dataset() -> list[dict[str, str]]:
    with open(DATA_DIR / "eval_dataset.json", encoding="utf-8") as f:
        return json.load(f)


def relevant_chunk_indices(chunks: list[str], substring: str) -> set[int]:
    needle = substring.lower()
    return {i for i, chunk in enumerate(chunks) if needle in chunk.lower()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    settings = get_settings()
    raw_text = (DATA_DIR / "sample_document.txt").read_text(encoding="utf-8")
    chunks = chunk_text(raw_text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    print(f"Sample document split into {len(chunks)} chunks "
          f"(chunk_size={settings.chunk_size_tokens}, overlap={settings.chunk_overlap_tokens})\n")

    embedder = load_embedder(settings.embedding_model_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = VectorStore(tmp_dir, "retrieval-eval")
        embeddings = embed_texts(chunks, embedder)
        ids = [f"sample-{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": "sample_document.txt", "document_id": "sample", "chunk_index": i}
            for i in range(len(chunks))
        ]
        store.add_chunks(ids, chunks, metadatas, embeddings)

        qa_pairs = load_eval_dataset()
        precisions: list[float] = []
        recalls: list[float] = []

        print(f"Evaluating retrieval on {len(qa_pairs)} Q/A pairs (top_k={args.top_k})\n")
        for item in qa_pairs:
            question = item["question"]
            substring = item["expected_source_substring"]
            relevant = relevant_chunk_indices(chunks, substring)
            if not relevant:
                print(f"WARNING: ground-truth substring not found in any chunk: {question!r}")
                continue

            query_embedding = embed_texts([question], embedder)[0]
            retrieved = store.query(query_embedding, args.top_k)
            retrieved_indices = [record["metadata"]["chunk_index"] for record in retrieved]

            hits = sum(1 for idx in retrieved_indices if idx in relevant)
            precision = hits / len(retrieved_indices) if retrieved_indices else 0.0
            recall = hits / len(relevant)
            precisions.append(precision)
            recalls.append(recall)

            status = "HIT " if hits else "MISS"
            print(f"[{status}] {question}")
            print(
                f"        precision@{args.top_k}={precision:.2f}  recall@{args.top_k}={recall:.2f}"
                f"  relevant_chunks={sorted(relevant)}  retrieved_chunks={retrieved_indices}"
            )

        if not precisions:
            print("\nNo evaluable Q/A pairs - check data/eval_dataset.json.")
            return 1

        mean_precision = sum(precisions) / len(precisions)
        mean_recall = sum(recalls) / len(recalls)
        print(f"\nMean precision@{args.top_k}: {mean_precision:.3f}")
        print(f"Mean recall@{args.top_k}:    {mean_recall:.3f}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
