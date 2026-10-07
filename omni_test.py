"""
FIRE-ICE — Nemotron Omni evaluation harness.

Sends pages of a PDF to NVIDIA's hosted Nemotron Nano Omni and saves what
it reads back, so you can compare its answers against the real pages
rather than judging from one browser session.

The question this answers: can a vision model recover the figures that
pypdf mangles — chart axes, multi-series plots, tables that straddle a
page break — and does it admit uncertainty instead of inventing numbers?

    export NVIDIA_API_KEY=nvapi-...        # from build.nvidia.com
    python omni_test.py --pdf data/markets/igu.pdf --pages 19 20 21

Writes omni/results.json and a readable omni/report.md.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = os.environ.get("OMNI_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")

# Inline base64 images have a size ceiling on this endpoint; stay well under it.
MAX_IMAGE_BYTES = 170_000

OUT_DIR = Path("omni")

# Four prompts, each testing something different. The last one matters most:
# a model that invents a plausible number is worse than one that declines.
PROMPTS = {
    "describe": (
        "Describe what is on this page: the headings, any charts, and any tables. "
        "Do not read values yet — just say what kinds of content are present."
    ),
    "chart": (
        "Read every chart on this page. For each one give: the title, the axis "
        "labels including units, the name of each data series, and the approximate "
        "value of each series at its first and last point. If a value is not "
        "legible, write UNCLEAR rather than estimating it."
    ),
    "table": (
        "Transcribe every table on this page as markdown. Preserve the column "
        "headers and units exactly as printed. If a table appears to continue "
        "beyond this page, say so. Do not fill in cells you cannot read — write "
        "UNCLEAR instead."
    ),
    "numbers": (
        "List every numeric value on this page that is a price, volume or "
        "capacity, with its unit and what it refers to. Quote the figure exactly "
        "as printed. If you are not certain a number belongs to the label you "
        "are pairing it with, say so explicitly."
    ),
}


def render_pages(pdf: Path, pages: list[int], dpi: int) -> list[tuple[int, Path]]:
    """pdftoppm is part of poppler — already present if you installed it earlier."""
    tmp = Path(tempfile.mkdtemp(prefix="omni_"))
    rendered = []

    try:
        import fitz
    except ImportError:
        fitz = None

    if fitz is not None:
        document = fitz.open(str(pdf))
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        for page in pages:
            if page < 1 or page > document.page_count:
                print(f"  ! page {page} is outside this PDF (1-{document.page_count})")
                continue
            path = tmp / f"p{page}.png"
            document[page - 1].get_pixmap(matrix=matrix).save(str(path))
            rendered.append((page, path))
        document.close()
        return rendered

    for page in pages:
        prefix = tmp / f"p{page}"
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-png", "-f", str(page), "-l", str(page),
             str(pdf), str(prefix)],
            check=True, capture_output=True,
        )
        matches = sorted(tmp.glob(f"p{page}*.png"))
        if matches:
            rendered.append((page, matches[0]))
        else:
            print(f"  ! page {page} did not render")
    return rendered


def encode(path: Path) -> str:
    """Base64 the image, shrinking until it fits the inline limit."""
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow is needed: pip install pillow")

    image = Image.open(path).convert("RGB")
    width = image.width
    smallest = None

    # Drop quality first, then resolution: text stays legible longer under
    # JPEG compression than under downscaling.
    for attempt in range(40):
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
            # Below this, chart labels stop being readable anyway.
            print(f"    ! could not fit under {MAX_IMAGE_BYTES // 1024} KB "
                  f"({len(smallest) // 1024} KB at {width}px) — sending anyway")
            return base64.b64encode(smallest).decode()
        width = int(width * 0.85)


def ask(image_b64: str, prompt: str, api_key: str, reasoning: bool) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
        "max_tokens": 2000,
        "temperature": 0.1,
    }
    if not reasoning:
        # The model reasons by default; this is the documented toggle.
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    started = time.time()
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        json=payload,
        timeout=180,
    )
    elapsed = round(time.time() - started, 1)

    if not response.ok:
        return {"ok": False, "seconds": elapsed,
                "error": f"HTTP {response.status_code}: {response.text[:400]}"}

    data = response.json()
    message = data.get("choices", [{}])[0].get("message", {})
    text = message.get("content") or message.get("reasoning_content") or ""
    usage = data.get("usage", {})

    return {
        "ok": True,
        "seconds": elapsed,
        "text": text.strip(),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="path to the PDF")
    parser.add_argument("--pages", type=int, nargs="+", required=True)
    parser.add_argument("--prompts", nargs="+", default=["chart", "table"],
                        choices=list(PROMPTS), help="which tests to run per page")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-reasoning", action="store_true",
                        help="disable the model's thinking mode")
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY is not set.\n"
            "Generate one at build.nvidia.com, then:\n"
            "  export NVIDIA_API_KEY=nvapi-..."
        )

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"No such PDF: {pdf}")

    print(f"Rendering {len(args.pages)} page(s) at {args.dpi} dpi...")
    rendered = render_pages(pdf, args.pages, args.dpi)
    if not rendered:
        raise SystemExit("Nothing rendered.")

    OUT_DIR.mkdir(exist_ok=True)
    results = []

    for page, image_path in rendered:
        encoded = encode(image_path)
        kb = len(encoded) * 3 // 4 // 1024
        print(f"\nPage {page}  ({kb} KB sent)")

        for name in args.prompts:
            print(f"  {name}...", end=" ", flush=True)
            outcome = ask(encoded, PROMPTS[name], api_key, not args.no_reasoning)
            if outcome["ok"]:
                unclear = outcome["text"].upper().count("UNCLEAR")
                print(f"{outcome['seconds']}s, {len(outcome['text'])} chars"
                      + (f", {unclear} UNCLEAR" if unclear else ""))
            else:
                print(outcome["error"][:120])

            results.append({"page": page, "prompt": name,
                            "question": PROMPTS[name], **outcome})

        # Keep a copy of exactly what the model saw, so you can check its reading.
        (OUT_DIR / f"page-{page}.jpg").write_bytes(base64.b64decode(encoded))

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Nemotron Omni — page reading test",
        "",
        f"Model: `{MODEL}`",
        f"Source: `{pdf}`",
        f"Reasoning: {'off' if args.no_reasoning else 'on'}",
        "",
        "Each page image is saved beside this file, so every answer can be",
        "checked against what the model actually saw.",
        "",
    ]
    for row in results:
        lines.append(f"## Page {row['page']} — {row['prompt']}")
        lines.append("")
        lines.append(f"*{row['question']}*")
        lines.append("")
        if row["ok"]:
            lines.append(f"`{row['seconds']}s · {row.get('completion_tokens')} tokens out`")
            lines.append("")
            lines.append(row["text"])
        else:
            lines.append(f"**Failed:** {row['error']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    (OUT_DIR / "report.md").write_text("\n".join(lines))

    failures = sum(1 for r in results if not r["ok"])
    print(f"\n{len(results)} call(s), {failures} failed")
    print(f"Wrote {OUT_DIR}/report.md and {OUT_DIR}/results.json")
    print("\nNow open the saved page images beside the report and check the")
    print("model's figures against the real ones. That comparison is the test.")


if __name__ == "__main__":
    main()
