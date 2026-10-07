"""
FIRE-ICE — retrieval.

Embeds a question in QUERY mode and returns the closest corpus chunks,
with optional metadata filters.

    python retrieve.py "Qatar expansion capacity" --k 5
    python retrieve.py "Henry Hub spread" --topic markets
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from core import INDEX_DIR, Chunk, embed

_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("faiss is needed: pip install faiss-cpu") from exc

    index_path = INDEX_DIR / "corpus.faiss"
    if not index_path.exists():
        raise SystemExit(f"No index at {index_path}. Run ingest.py first.")

    with open(INDEX_DIR / "chunks.pkl", "rb") as handle:
        rows = pickle.load(handle)

    _cache = {
        "index": faiss.read_index(str(index_path)),
        "chunks": [Chunk(**row) for row in rows],
        "manifest": json.loads((INDEX_DIR / "manifest.json").read_text()),
    }
    return _cache


def search(question: str, k: int = 6, **filters) -> list[tuple[Chunk, float]]:
    """
    Returns (chunk, score) pairs, best first. Filters are exact matches on
    metadata fields, e.g. topic="markets" or company="Cheniere".
    """
    store = _load()
    vector = np.asarray(embed([question], mode="query"), dtype="float32")
    vector /= max(float(np.linalg.norm(vector)), 1e-9)

    # Over-fetch when filtering, so the filter has candidates to work with.
    fetch = k * 6 if filters else k
    scores, ids = store["index"].search(vector, min(fetch, len(store["chunks"])))

    results: list[tuple[Chunk, float]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        chunk = store["chunks"][idx]
        if any(getattr(chunk, field, None) != value for field, value in filters.items() if value):
            continue
        results.append((chunk, float(score)))
        if len(results) >= k:
            break

    return results


def format_evidence(hits: list[tuple[Chunk, float]]) -> str:
    """Numbered passages the generator can cite as [S1], [S2]."""
    blocks = []
    for i, (chunk, score) in enumerate(hits, start=1):
        blocks.append(
            f"[S{i}] {chunk.citation()}  (similarity {score:.3f})\n"
            f"Source: {chunk.source}\n"
            f"{chunk.chunk_text}"
        )
    return "\n\n---\n\n".join(blocks) if blocks else "No matching passages were retrieved."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--topic")
    parser.add_argument("--company")
    parser.add_argument("--project")
    parser.add_argument("--region")
    args = parser.parse_args()

    hits = search(
        args.question, k=args.k,
        topic=args.topic, company=args.company,
        project=args.project, region=args.region,
    )
    for chunk, score in hits:
        print(f"{score:.3f}  {chunk.citation()}  [{chunk.chunk_id}]")
        print(f"        {chunk.chunk_text[:160].replace(chr(10), ' ')}...\n")
