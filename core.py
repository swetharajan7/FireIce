"""
FIRE-ICE — shared plumbing.

Config, the Nebius client, document loading, chunking and the metadata
schema. Both ingest.py and retrieve.py build on this.
"""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

import requests

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

NEBIUS_BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1")

# Where generation happens, surfaced so it can be seen rather than claimed.
# Retrieval is separate by design: the embedder runs on this machine.
INFERENCE_HOST = urlparse(NEBIUS_BASE_URL).netloc or NEBIUS_BASE_URL
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")

# Generation runs on Nebius (serverless, pay per token).
GEN_MODEL = os.environ.get("NEBIUS_GEN_MODEL", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")

# Embedding runs LOCALLY by default: Token Factory serves no Nemotron
# embedding model, and a 1B embedder is comfortable on a laptop.
#   local  -> sentence-transformers on this machine, free
#   nebius -> the /embeddings endpoint, for a model Nebius actually serves
EMBED_BACKEND = os.environ.get("FIREICE_EMBED_BACKEND", "local")
EMBED_MODEL = os.environ.get("FIREICE_EMBED_MODEL", "nvidia/llama-nemotron-embed-1b-v2")

# NVIDIA embedding models distinguish passage mode from query mode. Keeping
# them straight matters: indexing with the query prefix quietly degrades
# every search you will ever run against that index.
PASSAGE_PREFIX = os.environ.get("FIREICE_PASSAGE_PREFIX", "passage: ")
QUERY_PREFIX = os.environ.get("FIREICE_QUERY_PREFIX", "query: ")

DATA_DIR = Path(os.environ.get("FIREICE_DATA", "data"))
INDEX_DIR = Path(os.environ.get("FIREICE_INDEX", "index"))

CHUNK_CHARS = int(os.environ.get("FIREICE_CHUNK_CHARS", "1800"))
CHUNK_OVERLAP = int(os.environ.get("FIREICE_CHUNK_OVERLAP", "200"))

SYSTEM_PROMPT = """You are Fire-Ice, a natural-gas and LNG research assistant.

Distinguish facts, source-derived statements and inference. Prefer retrieved
domain evidence over unsupported prior knowledge. Cite source and date for
market claims, using the [S1], [S2] markers given with each passage.

Never invent prices, production figures, capacities or project status. If the
retrieved evidence does not answer the question, say so plainly and state what
document would be needed.

End every answer with a line in exactly this format, on its own line:

CONFIDENCE: high|medium|low — <one short clause on what limits it>

Use high only when the passages directly and fully answer the question; medium
when they answer it partly or require inference; low when you are largely
reasoning beyond the evidence."""


# --------------------------------------------------------------------------
# Metadata schema
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    document: str
    title: str
    source: str
    publication_date: str | None
    region: str | None
    company: str | None
    project: str | None
    topic: str | None
    page: int | None
    chunk_index: int
    chunk_text: str
    extra: dict = field(default_factory=dict)

    def citation(self) -> str:
        bits = [self.title or self.document]
        if self.publication_date:
            bits.append(self.publication_date)
        if self.page is not None:
            bits.append(f"p.{self.page}")
        return " — ".join(bits)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Nebius client
# --------------------------------------------------------------------------

def _headers() -> dict:
    if not NEBIUS_API_KEY:
        raise RuntimeError("NEBIUS_API_KEY is not set.")
    return {
        "Authorization": f"Bearer {NEBIUS_API_KEY}",
        "Content-Type": "application/json",
    }


_local_model = None


def _load_local():
    """Loads the embedder once, on the best device this Mac has."""
    global _local_model
    if _local_model is not None:
        return _local_model

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Local embedding needs: pip install sentence-transformers torch"
        ) from exc

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {EMBED_MODEL} on {device} (first run downloads the weights)...")
    _local_model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)
    return _local_model


def embed(texts: list[str], *, mode: str) -> list[list[float]]:
    """Embed a batch. mode is 'passage' when indexing, 'query' when searching."""
    if mode not in ("passage", "query"):
        raise ValueError("mode must be 'passage' or 'query'")

    prefix = PASSAGE_PREFIX if mode == "passage" else QUERY_PREFIX

    if EMBED_BACKEND == "local":
        model = _load_local()
        vectors = model.encode(
            [prefix + t for t in texts],
            batch_size=8,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    payload = {"model": EMBED_MODEL, "input": [prefix + t for t in texts]}

    response = requests.post(
        f"{NEBIUS_BASE_URL}/embeddings", headers=_headers(), json=payload, timeout=120
    )
    if not response.ok:
        raise RuntimeError(f"Embedding failed ({response.status_code}): {response.text[:500]}")

    data = response.json()
    rows = sorted(data["data"], key=lambda r: r["index"])
    return [r["embedding"] for r in rows]


def generate(system_prompt: str, user_prompt: str, *, max_tokens: int = 2000,
             model: str | None = None, effort: str | None = None) -> str:
    payload = {
        "model": model or GEN_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    if effort:
        payload["reasoning_effort"] = effort
    response = requests.post(
        f"{NEBIUS_BASE_URL}/chat/completions", headers=_headers(), json=payload, timeout=180
    )
    if not response.ok:
        raise RuntimeError(f"Generation failed ({response.status_code}): {response.text[:500]}")

    message = response.json()["choices"][0]["message"]
    return (message.get("content") or message.get("reasoning_content") or "").strip()


# --------------------------------------------------------------------------
# Loading and chunking
# --------------------------------------------------------------------------

def _read_pdf(path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is needed for PDFs: pip install pypdf") from exc

    reader = PdfReader(str(path))
    return [(i + 1, (page.extract_text() or "")) for i, page in enumerate(reader.pages)]


def _sidecar(path: Path) -> dict:
    """Optional <name>.meta.json sitting beside a document."""
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            print(f"  ! unreadable sidecar: {meta_path.name}")
    return {}


def split_text(text: str) -> list[str]:
    """Paragraph-aware chunking with overlap, so a sentence rarely splits."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= CHUNK_CHARS:
            current = f"{current}\n\n{para}" if current else para
            continue
        if current:
            chunks.append(current)
        # A single oversized paragraph gets hard-split.
        while len(para) > CHUNK_CHARS:
            chunks.append(para[:CHUNK_CHARS])
            para = para[CHUNK_CHARS - CHUNK_OVERLAP:]
        current = para
    if current:
        chunks.append(current)

    if CHUNK_OVERLAP and len(chunks) > 1:
        overlapped = [chunks[0]]
        for previous, nxt in zip(chunks, chunks[1:]):
            overlapped.append(previous[-CHUNK_OVERLAP:] + "\n\n" + nxt)
        chunks = overlapped

    return chunks


def load_chunks(data_dir: Path = DATA_DIR) -> Iterable[Chunk]:
    """
    Walks data/<topic>/... — the folder name becomes the default topic.
    Supports .md, .txt and .pdf. Metadata comes from an optional sidecar.
    """
    if not data_dir.exists():
        raise RuntimeError(f"No corpus at {data_dir.resolve()}")

    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in (".md", ".txt", ".pdf"):
            continue
        if path.name.endswith(".meta.json"):
            continue

        meta = _sidecar(path)
        relative = path.relative_to(data_dir)
        topic = meta.get("topic") or (relative.parts[0] if len(relative.parts) > 1 else None)

        if path.suffix.lower() == ".pdf":
            pages = _read_pdf(path)
        else:
            pages = [(None, path.read_text(errors="ignore"))]

        counter = 0
        for page_no, page_text in pages:
            for piece in split_text(page_text):
                yield Chunk(
                    chunk_id=f"{relative.as_posix()}#{counter}",
                    document=relative.as_posix(),
                    title=meta.get("title") or path.stem.replace("_", " "),
                    source=meta.get("source") or path.as_posix(),
                    publication_date=meta.get("publication_date"),
                    region=meta.get("region"),
                    company=meta.get("company"),
                    project=meta.get("project"),
                    topic=topic,
                    page=page_no,
                    chunk_index=counter,
                    chunk_text=piece,
                    extra={k: v for k, v in meta.items() if k not in {
                        "title", "source", "publication_date", "region",
                        "company", "project", "topic"}},
                )
                counter += 1
