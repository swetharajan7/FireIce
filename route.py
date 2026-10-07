"""
FIRE-ICE — model routing.

One question does not need one model. A definition lookup and a
multi-document comparison are different jobs, and the Nemotron family has a
variant sized for each.

    Lightning  fast recall: definitions, single figures, "what is X"
    Nano       short grounded answers where reasoning is light
    Super      the default: synthesis across several passages
    Ultra      hard work: comparisons, trade-offs, conflicting evidence,
               or thin retrieval where the reasoning has to carry more

Routing is decided by cheap signals — the shape of the question and the
strength of retrieval — not by another model call. It is deterministic,
inspectable and free.

Override any of it with environment variables:
    FIREICE_MODEL_LIGHTNING / _NANO / _SUPER / _ULTRA
    FIREICE_ROUTE=off        always use NEBIUS_GEN_MODEL
    FIREICE_ROUTE=ultra      always use Ultra, and so on
"""

from __future__ import annotations

import os
import re

MODELS = {
    "lightning": os.environ.get("FIREICE_MODEL_LIGHTNING", "nvidia/Nemotron-3_5-Lightning"),
    "nano": os.environ.get("FIREICE_MODEL_NANO", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"),
    "super": os.environ.get("FIREICE_MODEL_SUPER", "nvidia/nemotron-3-super-120b-a12b"),
    "ultra": os.environ.get("FIREICE_MODEL_ULTRA", "nvidia/Nemotron-3-Ultra-550b-a55b"),
}

# Token budget and reasoning effort per tier.
BUDGET = {
    "lightning": (1800, "none"),
    "nano": (1600, "low"),
    "super": (2400, "medium"),
    "ultra": (3200, "medium"),
}

# Questions that want a fact back, not an essay.
LOOKUP = re.compile(
    r"^\s*(what is|what's|who is|when did|when was|how much|how many|define|"
    r"list|name the|which year)\b", re.IGNORECASE)

# Questions that need several things held in mind at once.
HARD = re.compile(
    r"\b(compare|comparison|versus|vs\.?|trade-?off|trade-?offs|evaluate|"
    r"assess|implication|implications|why does|why do|why did|what would happen|"
    r"relationship between|drivers? of|conflict|disagree|contradict|"
    r"across|between .+ and |rank|prioriti[sz]e|recommend)\b", re.IGNORECASE)

# Questions that ask for judgement rather than retrieval.
JUDGEMENT = re.compile(
    r"\b(should|would you|best|worst|most important|risk|risks|outlook|"
    r"forecast|likely|strategy)\b", re.IGNORECASE)


def choose(question: str, hits=None, *, override: str | None = None) -> dict:
    """
    Returns {tier, model, max_tokens, effort, reason}.

    hits is the retrieval result (list of (chunk, score) pairs). Retrieval
    strength matters: when the passages are thin, the model has to reason
    harder about what it does and does not have, so routing moves up.
    """
    setting = (override or os.environ.get("FIREICE_ROUTE", "auto")).lower()

    if setting == "off":
        model = os.environ.get("NEBIUS_GEN_MODEL", MODELS["super"])
        return {"tier": "fixed", "model": model, "max_tokens": 2400,
                "effort": "medium", "reason": "Routing disabled."}

    if setting in MODELS:
        tokens, effort = BUDGET[setting]
        return {"tier": setting, "model": MODELS[setting], "max_tokens": tokens,
                "effort": effort, "reason": f"Forced to {setting} by FIREICE_ROUTE."}

    scores = [score for _, score in (hits or [])]
    top = max(scores) if scores else 0.0
    documents = len({chunk.document for chunk, _ in (hits or [])})

    words = len(question.split())
    is_lookup = bool(LOOKUP.match(question)) and words <= 14
    is_hard = bool(HARD.search(question))
    is_judgement = bool(JUDGEMENT.search(question))

    # Thin evidence is the strongest signal: the answer depends on reasoning
    # carefully about a weak match, which is exactly where small models slip.
    if top and top < 0.18:
        tier = "ultra"
        reason = f"Retrieval is weak (top {top:.3f}) — reasoning carries more of the answer."
    elif is_hard or (is_judgement and documents > 1):
        tier = "ultra"
        reason = "Comparison or judgement across sources."
    elif is_lookup and top >= 0.40:
        tier = "lightning"
        reason = "Direct lookup with a strong match."
    elif is_lookup:
        tier = "nano"
        reason = "Short factual question."
    elif words > 28 or documents > 2:
        tier = "super"
        reason = f"Synthesis across {documents} document(s)."
    else:
        tier = "super"
        reason = "Default: grounded synthesis."

    tokens, effort = BUDGET[tier]
    return {"tier": tier, "model": MODELS[tier], "max_tokens": tokens,
            "effort": effort, "reason": reason}


if __name__ == "__main__":
    class FakeChunk:
        def __init__(self, doc): self.document = doc

    examples = [
        ("What is JKM?", [(FakeChunk("a.pdf"), 0.55)]),
        ("What is the Waha hub price?", [(FakeChunk("a.pdf"), 0.36)]),
        ("Compare Henry Hub and TTF pricing dynamics", [(FakeChunk("a.pdf"), 0.48), (FakeChunk("b.pdf"), 0.44)]),
        ("How does boil-off gas affect carrier design?", [(FakeChunk("a.pdf"), 0.29)]),
        ("What drives LNG demand in power generation across Asia and Europe?",
         [(FakeChunk("a.pdf"), 0.45), (FakeChunk("b.pdf"), 0.42), (FakeChunk("c.pdf"), 0.40)]),
        ("Which LNG project should we prioritise?", [(FakeChunk("a.pdf"), 0.47), (FakeChunk("b.pdf"), 0.43)]),
    ]
    width = max(len(q) for q, _ in examples)
    for question, hits in examples:
        decision = choose(question, hits)
        print(f"{question:<{width}}  →  {decision['tier']:<9} {decision['reason']}")
