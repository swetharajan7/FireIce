"""
FIRE-ICE — corpus ingestion.

Walks the data directory, chunks every document, embeds each chunk in
PASSAGE mode, and writes a FAISS index plus a metadata sidecar.

    python ingest.py            # build or rebuild the index
    python ingest.py --dry-run  # chunk only, no embedding calls, no cost
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from core import INDEX_DIR, EMBED_MODEL, Chunk, embed, load_chunks

BATCH = 32


def build(dry_run: bool = False) -> None:
    chunks: list[Chunk] = list(load_chunks())
    if not chunks:
        print("No documents found. Add files under data/ and try again.")
        return

    docs = {c.document for c in chunks}
    print(f"{len(docs)} document(s) -> {len(chunks)} chunk(s)")
    for doc in sorted(docs):
        n = sum(1 for c in chunks if c.document == doc)
        print(f"  {doc}: {n} chunk(s)")

    if dry_run:
        print("\nDry run: nothing embedded, nothing spent.")
        print("Sample chunk:\n" + "-" * 60)
        print(chunks[0].chunk_text[:600])
        return

    print(f"\nEmbedding with {EMBED_MODEL} in passage mode...")
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        vectors.extend(embed([c.chunk_text for c in batch], mode="passage"))
        print(f"  {min(start + BATCH, len(chunks))}/{len(chunks)}")

    matrix = np.asarray(vectors, dtype="float32")

    # Cosine similarity via inner product on normalised vectors.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("faiss is needed: pip install faiss-cpu") from exc

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "corpus.faiss"))

    with open(INDEX_DIR / "chunks.pkl", "wb") as handle:
        pickle.dump([c.to_dict() for c in chunks], handle)

    (INDEX_DIR / "manifest.json").write_text(json.dumps({
        "embed_model": EMBED_MODEL,
        "dimensions": int(matrix.shape[1]),
        "chunks": len(chunks),
        "documents": sorted(docs),
    }, indent=2))

    print(f"\nIndexed {len(chunks)} chunks ({matrix.shape[1]} dimensions) into {INDEX_DIR}/")
    print("Reindex whenever you change the embedding model or the chunk settings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="chunk without embedding")
    args = parser.parse_args()
    build(dry_run=args.dry_run)
