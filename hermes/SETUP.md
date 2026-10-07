# FIRE-ICE + Hermes

Hermes is the control plane. FIRE-ICE is the research tool it calls.

```
          you ──► Hermes Agent  (terminal, Telegram, Slack, Discord)
                      │
      ┌───────────────┼────────────────┐
      │               │                │
  personal        FIRE-ICE          Tavily
  memory          MCP server        (built in,
  (Hermes)        ├ search          keyless)
                  ├ ask
                  ├ recall          generation
                  ├ remember        ──► Nemotron on
                  └ status              Nebius serverless
```

Three memories, three owners:

| Memory | Lives in | Holds |
|---|---|---|
| Personal | Hermes `USER.md` / `MEMORY.md` | preferences, how you want answers |
| Knowledge | FIRE-ICE index | your LNG corpus |
| Research | FIRE-ICE `memory/` | durable facts, past investigations |

Nothing in the corpus leaves your machine. Only the question and the
retrieved passages reach a model.

---

## 1. Install Hermes

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Free and open source — no Nous account or credits needed. You only need
those if you want Nous's own hosted models, which you do not: the track
requires an NVIDIA open-source model, and Nemotron is already yours.

---

## 2. Point it at Nemotron on Nebius

Nebius exposes an OpenAI-compatible endpoint, so Hermes treats it as a
custom OpenAI provider. Run the interactive setup first:

```bash
hermes
```

When it asks for a model provider, choose the OpenAI-compatible option and
give it:

```
Base URL : https://api.studio.nebius.com/v1
API key  : <your Nebius key>
Model    : nvidia/nemotron-3-super-120b-a12b
```

If the flow differs in your version, the same values can be set directly:

```bash
hermes config set llm.base_url https://api.studio.nebius.com/v1
hermes config set llm.model nvidia/nemotron-3-super-120b-a12b
hermes config set OPENAI_API_KEY <your Nebius key>
```

Check what the keys are actually called in your build before trusting
those three lines:

```bash
hermes config list
hermes --help
```

Verify:

```bash
hermes doctor
```

---

## 3. Give it Tavily (free)

Tavily is built into Hermes with keyless access, so web search costs you
nothing from your Tavily balance:

```bash
hermes config set web.backend tavily
```

To use your own key for higher limits instead:

```bash
hermes config set TAVILY_API_KEY tvly-your-key
```

---

## 4. Connect FIRE-ICE

FIRE-ICE already speaks MCP. Register it so Hermes can search your
library:

```bash
hermes mcp add fire-ice \
  --command /Users/sr/Desktop/fireice/.venv/bin/python \
  --args /Users/sr/Desktop/fireice/mcp_server.py \
  --cwd /Users/sr/Desktop/fireice
```

If `hermes mcp` is not the right subcommand in your build, check
`hermes --help` for how MCP servers are registered; the three values
needed are always the same — the venv's python, the server path, and the
project directory as the working directory.

Confirm the tools appear:

```bash
hermes
> What tools do you have?
```

You should see `fireice_search`, `fireice_ask`, `fireice_recall`,
`fireice_remember` and `fireice_status`.

---

## 5. Add the skills

Copy the skill files into Hermes's skills directory (check
`hermes config list` for its location — commonly `~/.hermes/skills/`):

```bash
cp hermes/skills/*.md ~/.hermes/skills/
```

Then:

```bash
hermes
> /lng-research How exposed is European supply to a Hormuz closure?
```

---

## 6. Always-on without a server

A laptop sleeps, so "always-on" needs something that runs without it.

**Scheduled tasks in Hermes** handle the local case — a morning briefing
that runs when the machine is awake.

**Nebius Serverless Jobs** handle the rest: a job wakes on a schedule,
runs the briefing, writes the result, and exits. You are billed for the
minutes it runs, not for an idle VM. That is the right shape for this,
and it keeps the $25 credit intact.

A dedicated VM is the wrong trade here: roughly $15–35 a month for an
always-idle machine, and it dissolves the claim that your corpus stays on
your own hardware.

---

## 7. Messaging channels (optional)

```bash
hermes gateway setup
hermes gateway start
```

Then Fire-Ice answers from Telegram or Signal, still reading a corpus that
never left your Mac. Good for the demo video: ask from your phone, watch
it cite your own PDFs.
