---
name: lng-research
description: Research an LNG or natural-gas question against the user's own document library, returning a cited synthesis with an honest account of what the sources do not cover.
---

# LNG research

Use this whenever the user asks a substantive question about LNG, natural
gas markets, projects, contracts, shipping or regulation.

## Procedure

1. **Recall first.** Call `fireice_recall` with the topic. If the user has
   investigated something related before, say so in one line before
   answering — it is the difference between a search engine and an agent
   that knows them.

2. **Search the library.** Call `fireice_search` with the question. Use
   `topic` to narrow when the question is clearly about one area
   (markets, lng_projects, contracts, shipping, regulation, companies,
   technology, your_notes).

3. **Read the similarity scores.** They are returned with each passage.
   Below about 0.20 means the library probably does not cover this. Say
   so rather than stretching weak passages into an answer.

4. **If the library is thin, offer the web.** Tavily is available and
   keyless. Ask before searching it, and keep the two kinds of evidence
   clearly separated in the answer — the user's own documents are not the
   same as a web result.

5. **Answer in three layers:**
   - the conclusion, stated plainly first
   - the evidence, citing `[S1]`, `[S2]` with document and page
   - what the sources do not cover, and what document would answer it

## Rules

- Never state a price, capacity, volume or project status that is not in
  a retrieved passage.
- Never expand an abbreviation unless the expansion appears in the
  passages. If you know what it stands for but the sources do not say,
  use the abbreviation alone.
- Distinguish what a source says from what you are inferring. Mark
  inference as inference.
- If two sources disagree, show both and say which is more recent.

## Remembering

If the conversation establishes something durable about the user's work —
a project they are tracking, a standing preference — call
`fireice_remember` with one short sentence. At most one per conversation,
and never to store the content of an answer.
