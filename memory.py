"""
FIRE-ICE — memory.

Three kinds, kept deliberately separate:

  USER.md         who you are, what you work on, how you want answers.
                  You write this. The agent only reads it.

  MEMORY.md       durable facts the agent has learned about your work.
                  The agent appends here, one line at a time, and only
                  when it says so explicitly. Narrow write path on purpose:
                  an unbounded memory fills with noise within a week.

  sessions.jsonl  every question and answer, embedded locally so past
                  investigations can be recalled by meaning rather than
                  by scrolling. This is the "leave, come back tomorrow,
                  it remembers what you were working on" layer.

Everything here stays on this machine.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core import embed

_PROJECT = Path(__file__).resolve().parent
MEMORY_DIR = Path(os.environ.get("FIREICE_MEMORY", _PROJECT / "memory"))
USER_FILE = MEMORY_DIR / "USER.md"
FACTS_FILE = MEMORY_DIR / "MEMORY.md"
SESSIONS_FILE = MEMORY_DIR / "sessions.jsonl"
SESSION_VECTORS = MEMORY_DIR / "sessions.npy"

MAX_FACTS = 120          # beyond this, the oldest are dropped
MAX_USER_CHARS = 4000    # keep the profile from crowding out evidence

# Appended to the system prompt when memory is in play.
MEMORY_RULES = """

You have access to a memory of this user and of your past work with them.
Treat it as context about who is asking, never as evidence about the world:
a market claim must still come from the retrieved passages.

If this conversation establishes a durable fact worth remembering — a project
the user is working on, a preference about how they want answers, a decision
they have made — end your reply with a line of exactly this form:

REMEMBER: <one short sentence>

Use it sparingly: at most one per answer, and only for things that will still
matter in a month. Do not use it to restate the question or the answer."""


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

USER_TEMPLATE = """# About me

Replace this with whatever Fire-Ice should know about you. For example:

- I work in LNG market analysis, focused on US Gulf Coast export economics.
- I care most about netback economics, basis differentials and cargo routing.
- Current projects: a confidence-scored hypothesis tracker; a regional
  benchmark comparison across nine hubs.

# How I want answers

- Lead with the conclusion, then the evidence.
- Always state what the sources do not cover.
- Use $/MMBtu unless the source uses another unit, and say which.
"""

FACTS_HEADER = """# Durable facts

Written by Fire-Ice as it learns them. Edit or delete any line freely —
this file is yours.
"""


def ensure_files() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not USER_FILE.exists():
        USER_FILE.write_text(USER_TEMPLATE)
        print(f"Created {USER_FILE} — worth editing before your next question.")
    if not FACTS_FILE.exists():
        FACTS_FILE.write_text(FACTS_HEADER)


# --------------------------------------------------------------------------
# Reading memory into a prompt
# --------------------------------------------------------------------------

def load_profile() -> str:
    ensure_files()
    text = USER_FILE.read_text().strip()
    return text[:MAX_USER_CHARS]


def load_facts(limit: int = 30) -> list[str]:
    ensure_files()
    lines = [
        line.strip() for line in FACTS_FILE.read_text().splitlines()
        if line.strip().startswith("- ")
    ]
    return lines[-limit:]


def context_block(question: str, recall: int = 3) -> str:
    """The memory section placed ahead of the retrieved passages."""
    parts = []

    profile = load_profile()
    if profile and "Replace this with" not in profile:
        parts.append(f"WHO IS ASKING:\n{profile}")

    facts = load_facts()
    if facts:
        parts.append("WHAT I HAVE LEARNED BEFORE:\n" + "\n".join(facts))

    past = search_sessions(question, k=recall)
    if past:
        lines = []
        for entry, score in past:
            when = entry.get("at", "")[:10]
            lines.append(f"- [{when}] asked: {entry['question']}\n  concluded: {entry['summary']}")
        parts.append("RELATED WORK FROM EARLIER SESSIONS:\n" + "\n".join(lines))

    if not parts:
        return ""

    return "MEMORY (context about the user, not evidence):\n\n" + "\n\n".join(parts)


# --------------------------------------------------------------------------
# Writing memory
# --------------------------------------------------------------------------

REMEMBER_PATTERN = re.compile(
    r"^[\s>*_\-]*\**\s*REMEMBER\s*\**\s*:\s*\**\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)


def split_remember(answer: str) -> tuple[str, str | None]:
    """Pulls a REMEMBER line off the answer so it is stored, not displayed."""
    match = REMEMBER_PATTERN.search(answer)
    if not match:
        return answer.strip(), None
    fact = match.group(1).strip().strip("*_ ")
    cleaned = (answer[:match.start()] + answer[match.end():]).strip()
    return cleaned, fact


def append_fact(fact: str) -> bool:
    """Adds a fact, skipping near-duplicates and trimming the oldest."""
    ensure_files()
    existing = load_facts(limit=MAX_FACTS)
    normalised = fact.lower().rstrip(".")
    if any(normalised in line.lower() for line in existing):
        return False

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(FACTS_FILE, "a") as handle:
        handle.write(f"- {fact}  ({stamp})\n")

    # Trim if the list has grown past the cap.
    lines = FACTS_FILE.read_text().splitlines()
    facts = [l for l in lines if l.strip().startswith("- ")]
    if len(facts) > MAX_FACTS:
        header = [l for l in lines if not l.strip().startswith("- ")]
        FACTS_FILE.write_text("\n".join(header + facts[-MAX_FACTS:]) + "\n")

    return True


def record_session(question: str, answer: str, meta: dict | None = None) -> None:
    """Appends the exchange and its embedding so it can be recalled later."""
    ensure_files()

    summary = " ".join(answer.split())[:400]
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "summary": summary,
        "meta": meta or {},
    }

    with open(SESSIONS_FILE, "a") as handle:
        handle.write(json.dumps(entry) + "\n")

    try:
        vector = np.asarray(embed([question], mode="passage"), dtype="float32")
        vector /= max(float(np.linalg.norm(vector)), 1e-9)
        if SESSION_VECTORS.exists():
            existing = np.load(SESSION_VECTORS)
            stacked = np.vstack([existing, vector])
        else:
            stacked = vector
        np.save(SESSION_VECTORS, stacked)
    except Exception as error:
        print(f"(session embedding failed, history still saved: {error})")


def search_sessions(question: str, k: int = 3, min_score: float = 0.35):
    """Finds past sessions related to this question, by meaning."""
    if not SESSIONS_FILE.exists() or not SESSION_VECTORS.exists():
        return []

    try:
        entries = [json.loads(line) for line in SESSIONS_FILE.read_text().splitlines() if line.strip()]
        vectors = np.load(SESSION_VECTORS)
    except Exception:
        return []

    if not entries or vectors.shape[0] == 0:
        return []

    # Keep the two in step even if one write failed partway.
    count = min(len(entries), vectors.shape[0])
    entries, vectors = entries[:count], vectors[:count]

    try:
        query = np.asarray(embed([question], mode="query"), dtype="float32")[0]
    except Exception:
        return []
    query /= max(float(np.linalg.norm(query)), 1e-9)

    scores = vectors @ query
    order = np.argsort(-scores)[:k]
    return [(entries[i], float(scores[i])) for i in order if scores[i] >= min_score]


# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect FIRE-ICE memory.")
    parser.add_argument("--recall", help="search past sessions")
    parser.add_argument("--forget", type=int, metavar="N",
                        help="delete fact number N (as shown by --facts)")
    parser.add_argument("--facts", action="store_true", help="list durable facts")
    args = parser.parse_args()

    ensure_files()

    if args.facts or args.forget is not None:
        facts = load_facts(limit=MAX_FACTS)
        if args.forget is not None:
            index = args.forget - 1
            if 0 <= index < len(facts):
                removed = facts.pop(index)
                FACTS_FILE.write_text(FACTS_HEADER + "\n" + "\n".join(facts) + "\n")
                print(f"Forgot: {removed}")
            else:
                print("No fact with that number.")
        else:
            if not facts:
                print("No durable facts yet.")
            for i, fact in enumerate(facts, start=1):
                print(f"{i:>3}. {fact[2:]}")

    elif args.recall:
        hits = search_sessions(args.recall, k=5, min_score=0.0)
        if not hits:
            print("No past sessions recorded yet.")
        for entry, score in hits:
            print(f"\n{score:.3f}  [{entry['at'][:10]}]  {entry['question']}")
            print(f"       {entry['summary'][:200]}...")

    else:
        facts = load_facts(limit=MAX_FACTS)
        sessions = SESSIONS_FILE.read_text().splitlines() if SESSIONS_FILE.exists() else []
        print(f"Profile : {USER_FILE}  ({len(load_profile())} chars)")
        print(f"Facts   : {len(facts)}")
        print(f"Sessions: {len(sessions)}")
