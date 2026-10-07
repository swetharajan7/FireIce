# FIRE-ICE — Your Natural Gas Research Agent

Retrieval-grounded answering over your own natural-gas corpus, using
NVIDIA models on Nebius Token Factory.

```
Nemotron Embed  understands similarity
FAISS           remembers and searches the embeddings
Nemotron Nano   reasons and writes
Hermes          orchestrates (later)
```

## Setup

```bash
pip install -r requirements.txt

export NEBIUS_API_KEY=...
export NEBIUS_BASE_URL=https://api.studio.nebius.com/v1
export NEBIUS_EMBED_MODEL=nvidia/llama-nemotron-embed-1b-v2
export NEBIUS_GEN_MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2
```

Confirm both model IDs against the Token Factory catalog before a long
ingest run. Use **public/serverless** endpoints — a dedicated endpoint
bills continuously whether or not you are using it.

## Corpus

```
data/
  lng_projects/  markets/     contracts/   shipping/
  regulation/    companies/   technology/  your_notes/
```

Drop in `.md`, `.txt` or `.pdf`. The folder name becomes the default
topic. Richer metadata goes in a sidecar next to the file:

`qatar_expansion.pdf` → `qatar_expansion.pdf.meta.json`

```json
{
  "title": "North Field East expansion",
  "source": "QatarEnergy press release",
  "publication_date": "2026-03-11",
  "region": "Middle East",
  "company": "QatarEnergy",
  "project": "North Field East",
  "topic": "lng_projects"
}
```

## Use

```bash
python ingest.py --dry-run     # chunk only: free, checks your corpus
python ingest.py               # build the index

python retrieve.py "Qatar expansion capacity" --k 5
python ask.py "What drives the Henry Hub to TTF spread?"
python ask.py "Qatar timeline" --topic lng_projects --json
```

## Passage mode vs query mode

NVIDIA embedding models treat documents and questions differently.
`ingest.py` always embeds in **passage** mode and `retrieve.py` in
**query** mode. Getting this backwards silently degrades every search,
so do not change it without reindexing.

## Next

1. Add 20–50 good documents and run `ingest.py`.
2. Write ~50 questions with known-good answers. Measure retrieval hit
   rate first — if the right passage is not retrieved, no amount of
   generator tuning will fix the answer.
3. Only then consider fine-tuning, and train whichever component the
   evaluation shows is actually failing.


## Use Fire-Ice from Claude (MCP)

Fire-Ice can expose your local library to Claude Desktop, Claude Code or any
MCP client. Claude then queries your corpus and cites your documents by page —
without the documents ever leaving this machine.

Tools provided:

| Tool | What it does |
|---|---|
| `fireice_search` | Semantic search over your library, returns passages with citations |
| `fireice_ask` | The full grounded pipeline, with both confidence scores |
| `fireice_recall` | Durable facts and related past sessions |
| `fireice_remember` | Store one durable fact |
| `fireice_status` | What is indexed, which models, where the data lives |

### Install

```bash
pip install "mcp>=1.2"
python mcp_server.py        # should start without error, then Ctrl-C
```

### Register with Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fire-ice": {
      "command": "/Users/YOU/Desktop/fireice/.venv/bin/python",
      "args": ["/Users/YOU/Desktop/fireice/mcp_server.py"],
      "cwd": "/Users/YOU/Desktop/fireice",
      "env": {
        "NEBIUS_API_KEY": "your-key",
        "NEBIUS_BASE_URL": "https://api.studio.nebius.com/v1",
        "NEBIUS_GEN_MODEL": "nvidia/nemotron-3-super-120b-a12b",
        "FIREICE_EMBED_MODEL": "nvidia/llama-nemotron-embed-1b-v2"
      }
    }
  }
}
```

Use the full path to the virtual environment's Python, and set `cwd` to the
project folder so the index and memory are found. Restart Claude Desktop.

### Register with Claude Code

```bash
claude mcp add fire-ice -- /Users/YOU/Desktop/fireice/.venv/bin/python \
  /Users/YOU/Desktop/fireice/mcp_server.py
```

### What moves and what does not

The client receives passages and citations. Your PDFs, the FAISS index and
`memory/` stay on disk. Retrieval runs on this machine's GPU. Only
`fireice_ask` calls a hosted model, and only with the question and the
retrieved passages.
