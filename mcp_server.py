"""
FIRE-ICE — MCP server.

Exposes your local corpus and memory as tools an MCP client (Claude Desktop,
Claude Code, Cursor) can call.

The point is what does NOT move: your documents, embeddings, index and memory
stay on this machine. The client receives passages and citations, never files.
Retrieval runs locally on this laptop's GPU; only what you choose to surface
leaves the process.

Run it directly to check it loads:

    python mcp_server.py

Then register it with a client (see README).
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from core import EMBED_MODEL, GEN_MODEL, INDEX_DIR, DATA_DIR
from memory import load_facts, search_sessions, append_fact
from retrieve import search as local_search

mcp = FastMCP("fire-ice")


def _corpus_summary() -> dict:
    chunks = 0
    documents: list[str] = []
    manifest = INDEX_DIR / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            chunks = data.get("chunks", 0)
            documents = data.get("documents", [])
        except Exception:
            pass
    return {"chunks": chunks, "documents": documents}


@mcp.tool()
def fireice_search(query: str, k: int = 6, topic: str = "") -> str:
    """
    Search the user's private natural-gas and LNG document library.

    Returns the passages that best match the query, each with its source
    document, page and similarity score. The documents themselves never
    leave the user's machine — only these passages are returned.

    Use this when the user asks about LNG, natural gas markets, projects,
    contracts, shipping or regulation and the answer should come from their
    own library rather than from the open web.

    Args:
        query: what to look for, phrased as a question or topic
        k: how many passages to return (1-12)
        topic: optional filter — markets, lng_projects, contracts, shipping,
               regulation, companies, technology, your_notes
    """
    k = max(1, min(int(k), 12))
    try:
        hits = local_search(query, k=k, topic=topic or None)
    except SystemExit as error:
        return f"No index available: {error}"
    except Exception as error:
        return f"Search failed: {error}"

    if not hits:
        return (
            "No passages matched. The library may not cover this topic — "
            "say so plainly rather than answering from general knowledge."
        )

    blocks = []
    for i, (chunk, score) in enumerate(hits, start=1):
        blocks.append(
            f"[S{i}] {chunk.citation()}  (similarity {score:.3f})\n"
            f"Document: {chunk.document}\n"
            f"{chunk.chunk_text}"
        )

    top = max(score for _, score in hits)
    header = (
        f"{len(hits)} passage(s) from the user's local library. "
        f"Top similarity {top:.3f}"
        + (" — weak match, treat with caution.\n" if top < 0.40 else ".\n")
    )
    return header + "\n\n---\n\n".join(blocks)


@mcp.tool()
def fireice_ask(question: str, k: int = 6) -> str:
    """
    Run the full FIRE-ICE pipeline: retrieve from the local library, then
    answer with a local-first grounded agent, returning the answer plus its
    evidence and two confidence scores.

    Prefer fireice_search when you want to reason over the passages yourself.
    Use this when the user explicitly wants Fire-Ice's own assessment, or
    wants the confidence scores.

    Args:
        question: the question to answer
        k: how many passages to ground the answer in (1-12)
    """
    from ask import ask as run_ask

    try:
        result = run_ask(question, k=max(1, min(int(k), 12)))
    except Exception as error:
        return f"Fire-Ice could not answer: {error}"

    rc = result["retrieval_confidence"]
    lines = [
        result["answer"],
        "",
        f"Retrieval confidence: {rc['label']} (top {rc['top']}, "
        f"{rc.get('sources_used', 0)} document(s)) — {rc['note']}",
    ]
    if result.get("agent_confidence"):
        lines.append(f"Agent confidence: {result['agent_confidence']}")

    lines.append("")
    lines.append("Evidence:")
    for item in result["evidence"]:
        lines.append(f"  [{item['marker']}] {item['citation']}  ({item['similarity']})")

    return "\n".join(lines)


@mcp.tool()
def fireice_recall(query: str = "") -> str:
    """
    Recall what Fire-Ice knows about this user: durable facts it has learned,
    and past research sessions related to a query.

    Use this at the start of a research conversation to pick up where the
    user left off, or when they refer to earlier work.

    Args:
        query: optional — find past sessions related to this topic. Leave
               empty to list the durable facts only.
    """
    parts = []

    facts = load_facts(limit=60)
    if facts:
        parts.append("What Fire-Ice has learned:\n" + "\n".join(facts))
    else:
        parts.append("No durable facts recorded yet.")

    if query:
        sessions = search_sessions(query, k=5, min_score=0.0)
        if sessions:
            lines = [
                f"- [{entry['at'][:10]}] {entry['question']}\n  {entry['summary'][:220]}"
                for entry, _ in sessions
            ]
            parts.append("Related past sessions:\n" + "\n".join(lines))
        else:
            parts.append("No related past sessions.")

    return "\n\n".join(parts)


@mcp.tool()
def fireice_remember(fact: str) -> str:
    """
    Store one durable fact about the user's work in Fire-Ice's memory.

    Use sparingly, and only for things that will still matter in a month:
    a project they are working on, a standing preference, a decision made.
    Never use it to store the content of an answer.

    Args:
        fact: one short sentence
    """
    fact = fact.strip()
    if not fact:
        return "Nothing to remember."
    if len(fact) > 240:
        return "Too long — a durable fact should be one short sentence."

    stored = append_fact(fact)
    return f"Remembered: {fact}" if stored else "Already known — nothing added."


@mcp.tool()
def fireice_status() -> str:
    """
    Describe the user's library: how many documents and passages are indexed,
    which models are configured, and where the data lives.

    Use this when the user asks what Fire-Ice can see, or before telling them
    their library does not cover something.
    """
    summary = _corpus_summary()
    facts = len(load_facts(limit=500))
    sessions_file = Path("memory/sessions.jsonl")
    sessions = len(sessions_file.read_text().splitlines()) if sessions_file.exists() else 0

    docs = "\n".join(f"  - {d}" for d in summary["documents"]) or "  (none indexed yet)"

    return (
        f"FIRE-ICE local library\n"
        f"  passages indexed : {summary['chunks']}\n"
        f"  durable facts    : {facts}\n"
        f"  past sessions    : {sessions}\n"
        f"  retriever        : {EMBED_MODEL} (runs locally)\n"
        f"  generator        : {GEN_MODEL}\n"
        f"  corpus path      : {DATA_DIR.resolve()}\n\n"
        f"Documents:\n{docs}\n\n"
        f"Documents, embeddings and memory stay on this machine."
    )


if __name__ == "__main__":
    mcp.run()
