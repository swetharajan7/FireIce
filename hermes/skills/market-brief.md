---
name: market-brief
description: Produce a dated natural-gas market brief from the user's library plus, if asked, current web evidence.
---

# Market brief

A short, dated briefing the user can read in two minutes.

## Procedure

1. Call `fireice_status` first. Know what the library actually covers and
   how current it is before writing anything.

2. Call `fireice_search` for each theme the brief needs: prices and
   benchmarks, supply and outages, demand, policy and sanctions, shipping
   and chokepoints. Separate searches retrieve better than one broad one.

3. Note the coverage boundary. If the library ends in June 2026 and today
   is October, say so at the top. A brief that silently presents stale
   figures as current is worse than no brief.

4. If the user wants current numbers, offer a Tavily search and mark
   those lines clearly as web evidence rather than library evidence.

## Shape

```
NATURAL GAS BRIEF — <date>
Coverage: <what the library spans, and where web evidence was used>

HEADLINE
  one sentence: the thing that matters most today

PRICES
  benchmark levels with their dates and sources

WHAT MOVED
  two or three developments, each cited

WHAT TO WATCH
  what would change the picture

GAPS
  what this brief could not establish, and what would be needed
```

## Rules

- Every number carries its date. A price without a date is useless.
- Never carry a figure forward as though it were current.
- Keep it under 400 words. A brief that needs scrolling is a report.
