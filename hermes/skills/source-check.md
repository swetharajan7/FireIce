---
name: source-check
description: Verify a specific numerical or factual claim against the user's library, and say plainly whether it is supported, contradicted, or absent.
---

# Source check

The user has a claim — from a colleague, an article, a model output — and
wants to know whether their own documents support it.

## Procedure

1. Restate the claim in one line, precisely. If it is vague ("prices are
   high"), ask what specific figure or assertion to check.

2. Call `fireice_search` with the claim's key terms, not the whole
   sentence. A claim about "Qatar's 2026 expansion capacity" searches
   better as "Qatar expansion capacity 2026".

3. Search again from a second angle if the first returns weak scores. A
   single miss is not evidence of absence.

4. Return one of four verdicts, and nothing fuzzier:

   - **Supported** — the passages state it. Quote and cite.
   - **Contradicted** — the passages say otherwise. Show both.
   - **Partly** — the passages support some of it. Say which part.
   - **Not in the library** — no passage covers it. Name the kind of
     document that would.

5. Never resolve a check from your own knowledge. The whole point is
   whether *their* sources support it. If you happen to know the answer
   and the library does not contain it, say exactly that.

## Rules

- A near-miss is not a hit. If the passage discusses the right topic but
  not the specific claim, that is "not in the library".
- Dates matter. A figure that was true in 2025 does not support a claim
  about 2026.
