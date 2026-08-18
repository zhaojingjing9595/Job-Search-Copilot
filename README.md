# Job-Search-Copilot

An agentic job-search assistant: a [LangGraph](https://github.com/langchain-ai/langgraph) agent, backed by Gemini, that searches live job postings, evaluates them against your profile, drafts a tailored CV profile for the good matches, and logs everything to a tracking Google Sheet — deciding what to do at each step itself, not via a hardcoded pipeline.

**Stack:** Python · LangChain / LangGraph · Gemini · Chroma · sentence-transformers · scikit-learn · MCP

## How it works

```
goal (e.g. "find backend jobs in Tel Aviv and evaluate matches")
        │
        ▼
   LangGraph agent loop (agent/graph.py)
        │
   ┌────┴─────────┬──────────────────┬──────────────┐
   ▼               ▼                  ▼              ▼
search_jobs     get_profile     draft_cv_profile   log_posting
(agent/tools.py — LLM decides which tool to call, when, and how many times)
```

Before the LLM ever sees a posting, a retrieval cascade narrows the field:

```
~50 scraped (JobSpy MCP) → cosine top 30 (Chroma bi-encoder) → cross-encoder top 10 → Gemini reasons over the 10
```

The cascade only decides *how many* candidates reach the model, never *which ones are matches*. The LLM does all the actual judging — including writing the `match_reasoning` that lands in the tracking sheet — and cover-letter/CV-profile drafting is RAG-grounded in your real CV bullets, never invented.

See [job-application-copilot-guide.md](job-application-copilot-guide.md) for the full phase-by-phase build rationale, and `plans/` for design notes on specific pieces (cascade design, MCP integration, RAG/eval expansion).

## Project layout

| Path | Purpose |
|---|---|
| `main.py` | CLI entrypoint — streams the agent's reasoning/tool calls live |
| `agent/graph.py` | The LangGraph loop (agent node ↔ tools node) |
| `agent/tools.py` | The four tools the LLM can call |
| `services/` | Profile setup, vector store, cross-encoder reranker, RAG CV-profile drafting, cascade orchestration |
| `integrations/` | JobSpy search + Google Sheets logging/auth |
| `core/` | MCP server registry, MCP client, logging, constants |
| `eval/` | Offline evaluation — IR ranking metrics, groundedness, golden-set labeling |
| `scripts/` | Dev tooling — scrape cache, vector-store inspector, cascade debug view |
| `job_spy_mcp/` | Local JobSpy MCP server (Node, cloned separately — see Setup) |

## Setup

**Requires Python 3.12** (not 3.13 — `torch==2.2.2` has no macOS x86_64 wheel for 3.13).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Also requires Node.js and `uv`/`uvx` (for the Sheets MCP server) on your `PATH`.

Clone the JobSpy MCP server into `job_spy_mcp/` — `core/mcp_config.py` expects it at `job_spy_mcp/jobspy-mcp-server/src/index.js`:

```bash
git clone https://github.com/borgius/jobspy-mcp-server.git job_spy_mcp/jobspy-mcp-server
```

Then follow that repo's own setup instructions ([borgius/jobspy-mcp-server](https://github.com/borgius/jobspy-mcp-server)) — `npm install` plus its Python/Docker JobSpy prerequisites.

### Environment variables (`.env`)

| Variable | Used for |
|---|---|
| `GEMINI_API_KEY` | Gemini chat + embedding calls |
| `GOOGLE_SHEETS_CLIENT_ID` / `GOOGLE_SHEETS_CLIENT_SECRET` | OAuth client for Sheets access |
| `GOOGLE_SHEET_ID` | Target tracking spreadsheet |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | Optional LangSmith tracing |

### One-time profile setup

Generate `profile.json` from your CV (PDF/DOCX/TXT) plus a few clarifying questions:

```bash
python -m services.profile
```

### Google Sheet tracking setup

Create a Desktop-app OAuth client (Sheets + Drive APIs enabled) and a sheet with a `Sheet1` tab headed `company, title, match_reasoning, status, link, date, cv_profile_path`. No separate command to run — the first time `python main.py` logs a posting, a browser opens for consent and the token is cached under `.google_tokens/` for future runs. Duplicate links are skipped automatically; set `DRY_RUN=1` to log without writing.

## Running it

```bash
python main.py --recursion-limit 40 "Search for at most 12 AI Full Stack Developer postings in Tel Aviv, Israel, from the last 24 hours. Evaluate only the top 5 by fit. Draft a customized CV profile for at most 2 postings that is a genuinely strong match. Log every evaluated posting with real match_reasoning, then stop."
python main.py                              # uses the default goal
python main.py --recursion-limit 10 "..."
```

This streams each agent reasoning step, tool call, and tool result live, then prints the agent's final summary.

## Evaluation

Offline, outside the agent loop, in `eval/`:

```bash
python -m eval.build_golden_set     # label ~40 real postings (0/1/2), gates everything below
python -m eval.ranking_eval         # precision@10 / nDCG@10 / MRR across BM25, cosine, cross-encoder, LLM-only
python -m eval.groundedness_eval    # RAG vs. no-RAG CV-profile groundedness, judged by Gemini
```

Results and write-ups live in `eval/results/`.

## Dev tooling

```bash
python -m scripts.inspect_vector_store   # dump what's indexed in Chroma
python -m scripts.dev_cache              # replay a cached JobSpy scrape instead of re-scraping
python -m scripts.cascade_debug          # log cosine vs. cross-encoder scores per candidate to a debug sheet tab
```
