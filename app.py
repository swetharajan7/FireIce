"""
FIRE-ICE — local web interface.

    python app.py

Opens at http://localhost:8000. Everything runs on this machine: the
embedder, the index, the corpus and the memory. Only the question and the
retrieved passages are sent to the generator.

Nothing is exposed beyond localhost.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import retrieve
from core import DATA_DIR, EMBED_MODEL, GEN_MODEL, INDEX_DIR
from memory import FACTS_FILE, FACTS_HEADER, load_facts, search_sessions

app = FastAPI(title="FIRE-ICE")

STATIC = Path(__file__).parent / "static"
TOPICS = ["lng_projects", "markets", "contracts", "shipping",
          "regulation", "companies", "technology", "your_notes"]

# Indexing runs in the background so the page stays responsive.
index_state = {"running": False, "message": "", "finished_at": None}


# --------------------------------------------------------------------------

def corpus_status() -> dict:
    documents = []
    if DATA_DIR.exists():
        for path in sorted(DATA_DIR.rglob("*")):
            if path.suffix.lower() in (".pdf", ".md", ".txt") and not path.name.endswith(".meta.json"):
                documents.append({
                    "name": path.name,
                    "topic": path.relative_to(DATA_DIR).parts[0] if len(path.relative_to(DATA_DIR).parts) > 1 else "",
                    "kb": round(path.stat().st_size / 1024),
                })

    chunks = 0
    manifest = INDEX_DIR / "manifest.json"
    if manifest.exists():
        try:
            chunks = json.loads(manifest.read_text()).get("chunks", 0)
        except Exception:
            pass

    sessions_file = Path("memory/sessions.jsonl")
    sessions = len(sessions_file.read_text().splitlines()) if sessions_file.exists() else 0

    return {
        "documents": documents,
        "chunks": chunks,
        "facts": len(load_facts(limit=500)),
        "sessions": sessions,
        "embed_model": EMBED_MODEL,
        "gen_model": GEN_MODEL,
        "indexing": index_state,
    }


def rebuild_index() -> None:
    """Re-embeds the corpus and swaps the index in. Runs off the request thread."""
    index_state.update({"running": True, "message": "Reading documents...", "finished_at": None})
    try:
        import ingest
        index_state["message"] = "Embedding locally — this can take a few minutes."
        ingest.build(dry_run=False)
        retrieve._cache = None          # force a reload on the next search
        index_state["message"] = "Index rebuilt."
    except Exception as error:
        index_state["message"] = f"Indexing failed: {error}"
    finally:
        index_state["running"] = False
        index_state["finished_at"] = time.time()


# --------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    k: int = 6
    remember: bool = True


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return corpus_status()


@app.post("/api/ask")
def ask_endpoint(request: AskRequest):
    if not request.question.strip():
        return JSONResponse({"error": "Ask something first."}, status_code=400)
    if index_state["running"]:
        return JSONResponse(
            {"error": "The index is still rebuilding. Try again in a moment."},
            status_code=409,
        )

    from ask import ask as run_ask

    try:
        result = run_ask(request.question, k=request.k, remember=request.remember)
        return result
    except Exception as error:
        return JSONResponse({"error": str(error)}, status_code=500)


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), topic: str = Form("your_notes")):
    if topic not in TOPICS:
        topic = "your_notes"

    target = DATA_DIR / topic
    target.mkdir(parents=True, exist_ok=True)

    saved = []
    for item in files:
        name = Path(item.filename).name
        if Path(name).suffix.lower() not in (".pdf", ".md", ".txt"):
            continue
        with open(target / name, "wb") as handle:
            shutil.copyfileobj(item.file, handle)
        saved.append(name)

    if saved and not index_state["running"]:
        threading.Thread(target=rebuild_index, daemon=True).start()

    return {"saved": saved, "topic": topic, "indexing": index_state}


@app.post("/api/reindex")
def reindex():
    if not index_state["running"]:
        threading.Thread(target=rebuild_index, daemon=True).start()
    return index_state


@app.get("/api/memory")
def memory_endpoint(q: str = ""):
    facts = [f[2:] if f.startswith("- ") else f for f in load_facts(limit=500)]
    recent = []
    if q:
        recent = [
            {"question": e["question"], "at": e["at"][:10], "score": round(s, 3)}
            for e, s in search_sessions(q, k=5, min_score=0.0)
        ]
    else:
        sessions_file = Path("memory/sessions.jsonl")
        if sessions_file.exists():
            lines = sessions_file.read_text().splitlines()[-5:]
            for line in reversed(lines):
                try:
                    entry = json.loads(line)
                    recent.append({"question": entry["question"], "at": entry["at"][:10]})
                except Exception:
                    continue
    return {"facts": facts, "recent": recent}


class ForgetRequest(BaseModel):
    index: int


@app.post("/api/forget")
def forget(request: ForgetRequest):
    facts = load_facts(limit=500)
    i = request.index
    if 0 <= i < len(facts):
        removed = facts.pop(i)
        FACTS_FILE.write_text(FACTS_HEADER + "\n" + "\n".join(facts) + "\n")
        return {"forgot": removed}
    return JSONResponse({"error": "No fact at that position."}, status_code=400)


if __name__ == "__main__":
    print("\nFIRE-ICE — http://localhost:8000")
    print("Corpus, index and memory stay on this machine.\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
