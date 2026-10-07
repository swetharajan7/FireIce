"""
FIRE-ICE — evaluation harness.

Runs a question set through the pipeline and records what actually happened:
which documents were retrieved, how close the match was, what the agent
claimed about its own confidence, and where those two disagree.

    python evaluate.py                          # all questions, current model
    python evaluate.py --retrieval-only         # free: no generation calls
    python evaluate.py --model nvidia/...       # tag the run with a model
    python evaluate.py --limit 5                # a quick sample

Writes eval/results_<label>.csv and prints a summary table.

Comparing two models:
    python evaluate.py --model nvidia/nemotron-3-super-120b-a12b --label super
    python evaluate.py --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B --label nano
    python evaluate.py --compare super nano
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

QUESTION_FILE = Path("eval/questions.json")
RESULTS_DIR = Path("eval")


# --------------------------------------------------------------------------
# Question set
# --------------------------------------------------------------------------

def load_questions() -> list[dict]:
    """
    eval/questions.json is a list of objects:

        {
          "id": 5,
          "question": "How do Henry Hub, TTF and JKM benchmarks differ?",
          "expect_document": "GasMarketReport",   optional, substring match
          "notes": "pricing section"               optional
        }

    expect_document is what makes retrieval hit rate measurable. Leave it
    out and the question still runs; it just will not be scored.
    """
    if not QUESTION_FILE.exists():
        raise SystemExit(
            f"No question set at {QUESTION_FILE}.\n"
            "Create it with: python evaluate.py --init"
        )
    return json.loads(QUESTION_FILE.read_text())


def write_starter_questions() -> None:
    """Writes the 20 LNG test questions as a starting point."""
    questions = [
        "What are the main stages of the LNG value chain, from natural gas production to delivery to the end customer?",
        "How does a liquefaction plant convert natural gas into LNG, and why is the gas cooled to approximately -162 C?",
        "What factors determine the nameplate capacity and actual utilization rate of an LNG export terminal?",
        "What is the difference between baseload LNG, peak-shaving LNG, and small-scale LNG projects?",
        "How do Henry Hub, TTF, and JKM pricing benchmarks differ, and where is each benchmark most relevant?",
        "What are the main pricing structures used in long-term LNG sales and purchase agreements, including oil-indexed and gas-indexed contracts?",
        "What is the difference between FOB and DES LNG contracts, and how do those terms affect shipping responsibility and commercial risk?",
        "What factors influence LNG shipping costs, including voyage distance, vessel size, boil-off gas, canal fees, and charter rates?",
        "How does boil-off gas affect LNG storage and transportation, and how is it typically managed on modern LNG carriers?",
        "What is the role of regasification terminals, and how do onshore terminals differ from floating storage and regasification units?",
        "How do seasonal demand patterns influence LNG flows between Asia, Europe, and the Atlantic Basin?",
        "What are the principal drivers of LNG demand in power generation, industry, residential heating, and transportation?",
        "How can LNG projects reduce lifecycle greenhouse-gas emissions, including methane leakage, liquefaction energy use, shipping emissions, and regasification?",
        "What are the main technical and commercial risks faced by a new LNG export project before final investment decision?",
        "How do feedgas availability, pipeline connectivity, and upstream gas supply affect the economics and reliability of an LNG project?",
        "What role do long-term offtake agreements play in financing LNG projects, and why are creditworthy buyers important?",
        "How do geopolitical disruptions, sanctions, canal closures, or regional conflicts affect LNG trade routes and market prices?",
        "What is the difference between brownfield and greenfield LNG development, and what are the main advantages and disadvantages of each?",
        "How are LNG terminals and projects evaluated for energy security, data sovereignty, cybersecurity, and operational resilience?",
        "If you were comparing two LNG projects, which technical, financial, contractual, environmental, and market metrics would you use?",
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    QUESTION_FILE.write_text(json.dumps(
        [{"id": i + 1, "question": q, "expect_document": "", "notes": ""}
         for i, q in enumerate(questions)],
        indent=2,
    ))
    print(f"Wrote {len(questions)} questions to {QUESTION_FILE}.")
    print("\nFill in expect_document for each one — a substring of the filename")
    print("you believe should be retrieved. That is what makes hit rate real.")
    print("Leave it blank for questions your corpus genuinely cannot answer;")
    print("those test whether the agent refuses, which matters just as much.")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2, "none": 0}


def stated_level(stated: str | None) -> str | None:
    if not stated:
        return None
    match = re.match(r"\s*(high|medium|low)", stated, re.IGNORECASE)
    return match.group(1).lower() if match else None


def run_one(entry: dict, k: int, retrieval_only: bool) -> dict:
    from retrieve import search
    from ask import retrieval_confidence

    question = entry["question"]
    started = time.time()

    hits = search(question, k=k)
    rc = retrieval_confidence(hits)

    documents = []
    for chunk, _ in hits:
        if chunk.document not in documents:
            documents.append(chunk.document)

    expect = (entry.get("expect_document") or "").strip()
    if expect:
        hit = any(expect.lower() in d.lower() for d in documents)
        top_hit = expect.lower() in documents[0].lower() if documents else False
    else:
        hit = top_hit = None

    row = {
        "id": entry.get("id"),
        "question": question,
        "expect_document": expect,
        "retrieved_documents": " | ".join(documents),
        "top_score": rc["top"],
        "mean_top3": rc["mean_top3"],
        "retrieval_confidence": rc["label"],
        "hit": hit,
        "hit_at_1": top_hit,
        "agent_confidence": None,
        "overconfident": None,
        "answer_chars": None,
        "seconds": None,
        "answer": "",
    }

    if retrieval_only:
        row["seconds"] = round(time.time() - started, 1)
        return row

    from ask import ask as ask_pipeline

    # memory off, so results stay comparable across runs
    result = ask_pipeline(question, k=k, remember=False)
    agent = stated_level(result.get("agent_confidence"))

    row["agent_confidence"] = agent
    row["answer"] = result["answer"]
    row["answer_chars"] = len(result["answer"])
    row["seconds"] = round(time.time() - started, 1)

    # The combination worth watching: the agent sure of itself while the
    # corpus barely matched.
    if agent and rc["label"]:
        row["overconfident"] = (
            CONFIDENCE_RANK.get(agent, 0) - CONFIDENCE_RANK.get(rc["label"], 0) >= 2
        )

    return row


def summarise(rows: list[dict], label: str) -> None:
    scored = [r for r in rows if r["hit"] is not None]
    answered = [r for r in rows if r["agent_confidence"]]

    print("\n" + "=" * 68)
    print(f"RUN: {label}    questions: {len(rows)}")
    print("=" * 68)

    if scored:
        hits = sum(1 for r in scored if r["hit"])
        at1 = sum(1 for r in scored if r["hit_at_1"])
        print(f"Retrieval hit rate      {hits}/{len(scored)}  ({hits / len(scored):.0%})")
        print(f"Hit at rank 1           {at1}/{len(scored)}  ({at1 / len(scored):.0%})")
    else:
        print("Retrieval hit rate      not scored (no expect_document set)")

    tops = [r["top_score"] for r in rows if r["top_score"] is not None]
    if tops:
        print(f"Top similarity          mean {sum(tops) / len(tops):.3f}   "
              f"min {min(tops):.3f}   max {max(tops):.3f}")

    counts = {}
    for r in rows:
        counts[r["retrieval_confidence"]] = counts.get(r["retrieval_confidence"], 0) + 1
    print("Retrieval confidence    " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))

    if answered:
        agent_counts = {}
        for r in answered:
            agent_counts[r["agent_confidence"]] = agent_counts.get(r["agent_confidence"], 0) + 1
        print("Agent confidence        " + ", ".join(f"{k}: {v}" for k, v in sorted(agent_counts.items())))

        over = [r for r in answered if r["overconfident"]]
        print(f"Overconfident answers   {len(over)}/{len(answered)}  "
              f"(agent high while retrieval low)")
        for r in over:
            print(f"    Q{r['id']}: {r['question'][:60]}...")

        times = [r["seconds"] for r in answered if r["seconds"]]
        if times:
            print(f"Seconds per question    mean {sum(times) / len(times):.1f}")


def compare(labels: list[str]) -> None:
    print("\n" + "=" * 68)
    print("COMPARISON")
    print("=" * 68)
    print(f"{'metric':<26}" + "".join(f"{l:>20}" for l in labels))

    loaded = {}
    for label in labels:
        path = RESULTS_DIR / f"results_{label}.csv"
        if not path.exists():
            raise SystemExit(f"No results for '{label}' at {path}")
        with open(path) as handle:
            loaded[label] = list(csv.DictReader(handle))

    def metric(name, fn):
        print(f"{name:<26}" + "".join(f"{fn(loaded[l]):>20}" for l in labels))

    def hit_rate(rows):
        scored = [r for r in rows if r["hit"] in ("True", "False")]
        if not scored:
            return "n/a"
        hits = sum(1 for r in scored if r["hit"] == "True")
        return f"{hits}/{len(scored)} ({hits / len(scored):.0%})"

    def over_rate(rows):
        answered = [r for r in rows if r["agent_confidence"]]
        if not answered:
            return "n/a"
        over = sum(1 for r in answered if r["overconfident"] == "True")
        return f"{over}/{len(answered)}"

    def mean_seconds(rows):
        vals = [float(r["seconds"]) for r in rows if r["seconds"]]
        return f"{sum(vals) / len(vals):.1f}" if vals else "n/a"

    def mean_len(rows):
        vals = [int(r["answer_chars"]) for r in rows if r["answer_chars"]]
        return f"{sum(vals) / len(vals):.0f}" if vals else "n/a"

    metric("retrieval hit rate", hit_rate)
    metric("overconfident", over_rate)
    metric("mean seconds", mean_seconds)
    metric("mean answer chars", mean_len)
    print("\nRetrieval is identical across models by design — the index does not")
    print("change. Differences in the other rows are the generator's doing.")


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="write a starter question set")
    parser.add_argument("--retrieval-only", action="store_true", help="skip generation (free)")
    parser.add_argument("--model", help="generator model id for this run")
    parser.add_argument("--label", help="name for the output file")
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--compare", nargs="+", metavar="LABEL")
    args = parser.parse_args()

    if args.init:
        write_starter_questions()
        return

    if args.compare:
        compare(args.compare)
        return

    if args.model:
        os.environ["NEBIUS_GEN_MODEL"] = args.model

    label = args.label or ("retrieval" if args.retrieval_only
                           else os.environ.get("NEBIUS_GEN_MODEL", "run").split("/")[-1])

    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]

    print(f"Running {len(questions)} question(s), k={args.k}"
          + ("  [retrieval only]" if args.retrieval_only else f"  model={os.environ.get('NEBIUS_GEN_MODEL', 'default')}"))

    rows = []
    for i, entry in enumerate(questions, start=1):
        print(f"  {i}/{len(questions)}  {entry['question'][:58]}...")
        try:
            rows.append(run_one(entry, args.k, args.retrieval_only))
        except Exception as error:
            print(f"      failed: {error}")
            rows.append({
                "id": entry.get("id"), "question": entry["question"],
                "expect_document": entry.get("expect_document", ""),
                "retrieved_documents": "", "top_score": None, "mean_top3": None,
                "retrieval_confidence": "error", "hit": None, "hit_at_1": None,
                "agent_confidence": None, "overconfident": None,
                "answer_chars": None, "seconds": None, "answer": f"ERROR: {error}",
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"results_{label}.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summarise(rows, label)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
