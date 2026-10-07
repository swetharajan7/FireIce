"""
FIRE-ICE — reading the figures.

Text extraction loses charts. A page whose argument lives in a plot comes
out of pypdf as a scatter of axis labels, so the index never sees it and
retrieval cannot find it. This reads those pages with a vision model and
writes what it saw as a sidecar markdown file beside the PDF.

Because ingestion already walks data/ and indexes .md files, the next
`python ingest.py` picks the sidecar up and the figures become searchable
alongside the text. Nothing else in the pipeline changes.

    export NVIDIA_API_KEY=nvapi-...
    python vision.py --pdf data/markets/igu.pdf --pages 19 20 21
    python ingest.py

Provenance is the point here. Every line the model produces is marked as
model-read and unverified, with its page number, so an answer grounded in
a figure can always be traced back to the page and checked by eye. A
transcription is evidence about what a model saw, not the document itself.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API_URL = os.environ.get(
    "OMNI_BASE_URL", "https://integrate.api.nvidia.com/v1") + "/chat/completions"
MODEL = os.environ.get("OMNI_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
MAX_IMAGE_BYTES = 170_000

# One prompt, written for retrieval rather than for conversation: the output
# has to read like a passage, because that is what it becomes.
PROMPT = """You are transcribing one page of a technical report so its
figures can be searched later.

Write plain prose and markdown tables. Do not address the reader, do not
summarise, do not interpret. Record only what is printed on the page.

For each chart:
- its title, exactly as printed
- what the axes measure, with units, and whether a scale is logarithmic
- each data series by name
- the values you can read with confidence, each tied to its series and its
  point on the x-axis

For each table: transcribe it as markdown with the headers and units as
printed. Say if it appears to continue beyond this page.

Rules that matter more than completeness:
- Never estimate. If a value is not legible, write UNCLEAR.
- Never pair a label with a marker unless the connection is unambiguous.
- Never add context from your own knowledge. If the page says only
  "Q2 2026", do not expand it.
- If the page has no chart and no table, write exactly: NO FIGURES.
"""


def render(pdf: Path, pages: list[int], dpi: int) -> list[tuple[int, Path]]:
    try:
        import fitz
    except ImportError:
        raise SystemExit("PyMuPDF is needed: pip install pymupdf")

    tmp = Path(tempfile.mkdtemp(prefix="fireice_vision_"))
    document = fitz.open(str(pdf))
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)

    out = []
    for page in pages:
        if page < 1 or page > document.page_count:
            print(f"  ! page {page} is outside this PDF (1-{document.page_count})")
            continue
        path = tmp / f"p{page}.png"
        document[page - 1].get_pixmap(matrix=matrix).save(str(path))
        out.append((page, path))

    document.close()
    return out


def encode(path: Path) -> str:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width = image.width
    smallest = None

    for _ in range(40):
        resized = image if width >= image.width else image.resize(
            (width, max(1, int(image.height * width / image.width))), Image.LANCZOS)
        for quality in (85, 70, 55, 45):
            buffer = io.BytesIO()
            resized.save(buffer, "JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()
            if len(data) <= MAX_IMAGE_BYTES:
                return base64.b64encode(data).decode()
            smallest = data
        if width <= 700:
            return base64.b64encode(smallest).decode()
        width = int(width * 0.85)
    return base64.b64encode(smallest).decode()


def read_page(image_b64: str, api_key: str, reasoning: bool = True) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
        "max_tokens": 2500,
        "temperature": 0.1,
    }
    if not reasoning:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    started = time.time()
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            json=payload, timeout=180,
        )
    except requests.RequestException as error:
        return {"ok": False, "error": str(error), "seconds": round(time.time() - started, 1)}

    elapsed = round(time.time() - started, 1)
    if not response.ok:
        return {"ok": False, "seconds": elapsed,
                "error": f"HTTP {response.status_code}: {response.text[:300]}"}

    message = response.json().get("choices", [{}])[0].get("message", {})
    text = (message.get("content") or message.get("reasoning_content") or "").strip()
    return {"ok": True, "seconds": elapsed, "text": text}


def sidecar_path(pdf: Path) -> Path:
    """Beside the PDF, so ingestion inherits the same topic folder."""
    return pdf.with_name(pdf.stem + "-figures.md")


def load_existing(path: Path) -> dict[int, str]:
    """Pages already read, so a rerun costs nothing for them."""
    if not path.exists():
        return {}
    pages: dict[int, str] = {}
    current = None
    buffer: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("## Page "):
            if current is not None:
                pages[current] = "\n".join(buffer).strip()
            try:
                current = int(line.removeprefix("## Page ").split()[0])
            except ValueError:
                current = None
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        pages[current] = "\n".join(buffer).strip()
    return pages


def write_sidecar(path: Path, pdf: Path, pages: dict[int, str]) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"# Figures read from {pdf.name}",
        "",
        f"Source document: `{pdf.name}`  ",
        f"Read by: `{MODEL}` on {stamp}  ",
        "",
        "> These transcriptions were produced by a vision model reading page",
        "> images. They are evidence of what the model saw, not the document",
        "> itself. Every figure cited from here should be checked against the",
        "> page before it is relied on. Values the model could not read are",
        "> marked UNCLEAR.",
        "",
    ]
    for page in sorted(pages):
        body = pages[page].strip()
        if not body or body.upper().startswith("NO FIGURES"):
            continue
        lines.append(f"## Page {page}")
        lines.append("")
        lines.append(f"From {pdf.name}, page {page}, read by {MODEL}.")
        lines.append("")
        lines.append(body)
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read charts and tables from PDF pages into a searchable sidecar.")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--pages", type=int, nargs="+",
                        help="pages to read (1-based); omit with --all")
    parser.add_argument("--all", action="store_true", help="read every page")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--redo", action="store_true",
                        help="re-read pages already in the sidecar")
    parser.add_argument("--no-reasoning", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY is not set.\n"
            "Get a free key at build.nvidia.com, then:\n"
            "  export NVIDIA_API_KEY=nvapi-..."
        )

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"No such PDF: {pdf}")

    if args.all:
        import fitz
        document = fitz.open(str(pdf))
        pages = list(range(1, document.page_count + 1))
        document.close()
    elif args.pages:
        pages = args.pages
    else:
        raise SystemExit("Give --pages 19 20 21, or --all.")

    sidecar = sidecar_path(pdf)
    existing = {} if args.redo else load_existing(sidecar)
    todo = [p for p in pages if p not in existing]

    if existing:
        print(f"{len(existing)} page(s) already read; {len(todo)} to do.")
    if not todo:
        print("Nothing to read. Use --redo to read them again.")
        return

    print(f"Reading {len(todo)} page(s) with {MODEL}...")
    results = dict(existing)
    failures = 0

    for page, image in render(pdf, todo, args.dpi):
        print(f"  page {page}...", end=" ", flush=True)
        outcome = read_page(encode(image), api_key, not args.no_reasoning)
        if outcome["ok"]:
            text = outcome["text"]
            unclear = text.upper().count("UNCLEAR")
            empty = text.upper().startswith("NO FIGURES")
            print(f"{outcome['seconds']}s, "
                  + ("no figures" if empty else f"{len(text)} chars"
                     + (f", {unclear} UNCLEAR" if unclear else "")))
            results[page] = text
        else:
            failures += 1
            print(outcome["error"][:120])

    write_sidecar(sidecar, pdf, results)

    kept = sum(1 for t in results.values()
               if t.strip() and not t.upper().startswith("NO FIGURES"))
    print(f"\nWrote {sidecar}")
    print(f"{kept} page(s) with figures, {failures} failed")
    print("\nCheck the transcriptions against the real pages before trusting them.")
    print("Then run:  python ingest.py")


if __name__ == "__main__":
    main()
