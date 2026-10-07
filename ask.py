"""
FIRE-ICE — grounded answering.

Retrieve, then generate with Nemotron Nano over the retrieved passages only.
Prints three layers: Answer, Evidence, Research trail.

    python ask.py "What is driving the Henry Hub to TTF spread?"
    python ask.py "Qatar expansion timeline" --topic lng_projects --json
"""

from __future__ import annotations

import argparse
import json

import re

from core import SYSTEM_PROMPT, GEN_MODEL, EMBED_MODEL, INFERENCE_HOST, generate
from retrieve import search, format_evidence
from route import choose as choose_model
from memory import (
    MEMORY_RULES,
    append_fact,
    context_block,
    record_session,
    split_remember,
)


def retrieval_confidence(hits) -> dict:
    """
    How well the corpus covers the question, computed from similarity
    scores alone. This is arithmetic, not a model opinion: it says how
    close the retrieved passages were, not whether the answer is right.
    """
    if not hits:
        return {"label": "none", "top": None, "mean_top3": None,
                "note": "Nothing was retrieved."}

    scores = [score for _, score in hits]
    top = max(scores)
    mean_top3 = sum(scores[:3]) / min(3, len(scores))

    if top >= 0.55 and mean_top3 >= 0.45:
        label, note = "high", "Close matches across several passages."
    elif top >= 0.40:
        label, note = "medium", "Reasonable match, but not a direct one."
    else:
        label, note = "low", "No passage matched closely; the corpus may not cover this."

    return {
        "label": label,
        "top": round(top, 4),
        "mean_top3": round(mean_top3, 4),
        "sources_used": len({chunk.document for chunk, _ in hits}),
        "note": note,
    }


def split_confidence(answer: str) -> tuple[str, str | None]:
    """
    Pulls the agent's own confidence line off the end of the answer.

    Models wrap it in markdown in practice: "**CONFIDENCE:** medium",
    "*Confidence*: high", "- CONFIDENCE: low". The pattern tolerates
    leading bullets, asterisks and underscores on either side of the label.
    """
    match = re.search(
        r"^[\s>*_\-]*\**\s*CONFIDENCE\s*\**\s*:\s*\**\s*(.+)$",
        answer,
        re.MULTILINE | re.IGNORECASE,
    )
    if not match:
        return answer.strip(), None
    stated = match.group(1).strip().strip("*_ ")
    cleaned = answer[:match.start()].rstrip()
    return cleaned, stated


def ask(question: str, k: int = 6, remember: bool = True,
        model: str | None = None, **filters) -> dict:
    hits = search(question, k=k, **filters)
    evidence = format_evidence(hits)

    # Memory is context about the user, placed ahead of the evidence.
    memory = context_block(question) if remember else ""

    user_prompt = (
        (f"{memory}\n\n" if memory else "")
        + f"QUESTION:\n{question}\n\n"
        + f"RETRIEVED PASSAGES:\n\n{evidence}\n\n"
        + "Answer the question using these passages. Cite them as [S1], [S2] and so on. "
        + "State plainly where the passages do not cover the question."
    )

    system = SYSTEM_PROMPT + (MEMORY_RULES if remember else "")

    # The question and the strength of retrieval decide which Nemotron runs.
    routing = choose_model(question, hits, override=model)

    raw = generate(
        system, user_prompt,
        max_tokens=routing["max_tokens"],
        model=routing["model"],
        effort=routing["effort"],
    )
    raw, new_fact = split_remember(raw)
    stored_fact = append_fact(new_fact) if (remember and new_fact) else False

    answer, stated_confidence = split_confidence(raw)

    result = {
        "question": question,
        "answer": answer,
        "agent_confidence": stated_confidence,
        "retrieval_confidence": retrieval_confidence(hits),
        "evidence": [
            {
                "marker": f"S{i}",
                "citation": chunk.citation(),
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "similarity": round(score, 4),
                "excerpt": chunk.chunk_text[:400],
            }
            for i, (chunk, score) in enumerate(hits, start=1)
        ],
        "trail": {
            "retriever": EMBED_MODEL,
            "retrieval_ran_on": "this machine (local GPU)",
            "generator": routing["model"],
            "inference_host": INFERENCE_HOST,
            "tier": routing["tier"],
            "routing_reason": routing["reason"],
            "passages_retrieved": len(hits),
            "memory_used": bool(memory),
            "remembered": new_fact if stored_fact else None,
            "filters": {k2: v for k2, v in filters.items() if v},
        },
    }

    if remember:
        record_session(question, result["answer"], meta={
            "generator": routing["model"],
            "tier": routing["tier"],
            "routing_reason": routing["reason"],
            "retrieval_confidence": result["retrieval_confidence"]["label"],
        })

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--topic")
    parser.add_argument("--company")
    parser.add_argument("--project")
    parser.add_argument("--region")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--model", choices=["lightning", "nano", "super", "ultra"],
                        help="force a tier instead of routing")
    parser.add_argument("--no-memory", action="store_true",
                        help="skip memory entirely (used by the eval harness)")
    args = parser.parse_args()

    result = ask(
        args.question, k=args.k, remember=not args.no_memory, model=args.model,
        topic=args.topic, company=args.company,
        project=args.project, region=args.region,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n=== ANSWER ===\n")
        print(result["answer"])
        print("\n=== CONFIDENCE ===\n")
        rc = result["retrieval_confidence"]
        print(f"Retrieval : {rc['label']}  (top {rc['top']}, mean top-3 {rc['mean_top3']}, "
              f"{rc.get('sources_used', 0)} document(s))")
        print(f"            {rc['note']}")
        if result["agent_confidence"]:
            print(f"Agent     : {result['agent_confidence']}")

        print("\n=== EVIDENCE ===\n")
        for item in result["evidence"]:
            print(f"[{item['marker']}] {item['citation']}  ({item['similarity']})")
            print(f"      {item['source']}")
        if result["trail"].get("remembered"):
            print(f"\n=== REMEMBERED ===\n\n{result['trail']['remembered']}")

        print("\n=== ROUTING ===\n")
        t = result["trail"]
        print(f"{t.get('tier', '?')} \u2192 {t.get('generator', '?')}")
        print(f"            via {t.get('inference_host', '?')}")
        print(f"            {t.get('routing_reason', '')}")
        print(f"\nretrieval  {t.get('retriever', '?')}")
        print(f"            {t.get('retrieval_ran_on', '')}")

        print("\n=== RESEARCH TRAIL ===\n")
        print(json.dumps(result["trail"], indent=2))
