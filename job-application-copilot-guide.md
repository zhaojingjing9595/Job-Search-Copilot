# Job Application Copilot — Build Guide

An agentic app that searches job postings, evaluates them against your profile, drafts cover letters for good matches, and logs everything to a tracking sheet — with an LLM (Gemini) deciding what to do next at each step, not a hardcoded pipeline.

**Stack:** Python, LangChain, LangGraph, Gemini (free tier), Chroma, sentence-transformers, scikit-learn

**Scope note:** this started as "build an agent that hosts multiple MCP servers." It has since broadened into a closing study project, so it now also demonstrates vector search, RAG, deep-learning inference, classical ML, and evaluation. The agent loop is unchanged by that expansion — the additions sit *in front of* the LLM (narrowing what it sees) or *beside* it (measuring what it does). Detailed rationale for each addition lives in `plans/rag-ml-eval-expansion.md`.

---

## Architecture Overview

```
Goal (e.g. "find backend jobs in Tel Aviv and evaluate matches")
        │
        ▼
   LangGraph Agent Loop
        │
   ┌────┴─────┬──────────────┬──────────────────┬───────────────┐
   ▼           ▼              ▼                  ▼               ▼
search_jobs  get_profile   draft_cover_letter  log_posting  (LLM reasoning
  tool         tool          tool (RAG)          tool        between calls)
   │                            │
   │                            └── retrieves CV bullets from Chroma
   ▼
retrieval cascade (Phases 2.5–2.6), before the LLM sees anything:

  50 scraped ──cosine──▶ 30 ──cross-encoder──▶ 10 ──▶ LLM reasons & decides
                (bi-encoder)   (transformer)
```

The LLM decides which tool to call, when to call it, whether a posting is a match, and when it's done — that decision-making is the "agentic" part. The cascade only controls *how many* candidates reach it, never which ones are matches.

---

## Topic Coverage Map

| Topic | Where it lives |
|---|---|
| LLM | Gemini via `langchain-google-genai` (Phase 0) |
| Agent | LangGraph loop (Phase 5) |
| MCP | JobSpy server + Sheets tools (Phases 2–3) |
| Vector DB | Chroma, two collections (Phase 2.5) |
| NLP | chunking, embeddings, semantic dedup (Phase 2.5) |
| Deep learning | cross-encoder reranker, local transformer inference (Phase 2.6) |
| RAG | grounded cover-letter drafting (Phase 4.5) |
| Evaluation — IR | precision@k / nDCG over a golden set (Phase 8) |
| Evaluation — LLM | groundedness via LLM-as-judge (Phase 8) |
| Classical ML | clustering + logistic regression (Phase 9, optional) |
| Fine-tuning | cross-encoder fine-tune, expected negative result (Phase 10, optional) |

---

## Phase 0 — Setup

- [ ] Create a **dedicated** Google Cloud project (keep it separate from any billed project — enabling billing deletes the free tier on that project)
- [ ] Get a Gemini API key from Google AI Studio
- [ ] Sign up for a free job-board API key (e.g. Adzuna)
- [ ] Set up project structure:
  ```
  job-copilot/
  ├── main.py
  ├── .env
  ├── requirements.txt
  ├── profile.yaml
  ├── vector_store/          # Chroma persistence (Phase 2.5)
  ├── eval/                  # offline, outside the agent loop (Phase 8)
  │   ├── golden_set.json
  │   └── ranking_eval.py
  └── tools/
      ├── jobs.py
      ├── profile.py
      ├── sheets.py
      ├── vector_store.py    # Phase 2.5
      └── reranker.py        # Phase 2.6
  ```
- [ ] Install core dependencies:
  ```
  pip install langchain langchain-google-genai langgraph python-dotenv gspread requests
  ```
- [ ] Install the retrieval / ML dependencies (needed from Phase 2.5 onward — no extra API key, embeddings reuse the Gemini credential):
  ```
  pip install chromadb sentence-transformers scikit-learn umap-learn matplotlib rank-bm25
  ```

**Milestone:** a script that sends one hardcoded prompt to Gemini via `langchain-google-genai` and prints the response.

---

## Phase 1 — Profile Setup (one-time, CV-driven)

Instead of hand-writing the profile file, generate it once from your CV plus a few clarifying questions. This is a **separate one-time setup script** (`setup_profile.py`), not part of the live agent loop — you run it once during setup and edit the resulting YAML by hand afterward if anything changes.

**Step 1 — Extract text from your CV PDF**
- [ ] `pip install pypdf` (or `pdfplumber` for better layout handling)
- [ ] Write a function that reads the PDF and pulls out raw text

**Step 2 — Use the LLM to structure it (one-off call, not a `@tool`)**
- [ ] Send the extracted CV text to Gemini, asking it to extract structured fields (skills, experience summary, education, etc.)
- [ ] Request structured JSON output matching the shape you want — Gemini supports requesting a JSON schema directly, worth using here instead of hoping the model formats things correctly on its own

**Step 3 — Ask clarifying questions (no LLM needed)**
- [ ] Simple `input()` prompts for the things a CV won't contain: target roles, must-haves, dealbreakers, preferred locations, salary floor if relevant
- [ ] Merge these answers into the structured data from Step 2

**Step 4 — Save the merged result**
- [ ] Write the combined dict to `profile.yaml` — this is the same file your `get_profile()` tool reads later, so everything downstream (Phase 4 onward) works the same regardless of how the file was generated
- [ ] Write a plain `get_profile()` function (used later in Phase 4) that loads and returns this YAML as structured data
- [ ] Test `get_profile()` standalone (no LLM involved at agent-runtime)

**Milestone:** running `python setup_profile.py --cv my_resume.pdf`, answering a few prompts, and ending up with a populated, inspectable `profile.yaml`. Separately, `get_profile()` returns that data cleanly when called.

---

## Phase 2 — Job Board Tool (JobSpy MCP)

read on job board sourcing mcp server: https://www.remoet.dev/blog/mcp-servers-for-job-search-compared

**Why the change:** Adzuna doesn't cover Israel. JobSpy scrapes multiple boards (LinkedIn, Indeed, Glassdoor, Google Jobs, and others) and its supported country list explicitly includes Israel, so it fixes the geography problem. Trade-off: it's scraping-based rather than an official API, so it's more prone to rate-limiting/blocking (e.g. Cloudflare/CAPTCHA) than a clean REST API like Adzuna.

- [ ] Clone a JobSpy MCP server implementation, e.g.:
  ```
  git clone https://github.com/lockie/jobspy-mcp
  cd jobspy-mcp
  uv run jobspy-mcp
  ```
  (or `borgius/jobspy-mcp-server` if you prefer the Node/Docker-based variant)
- [ ] Run it locally and confirm it starts (`npx @modelcontextprotocol/inspector` or equivalent works for a quick manual check)
- [ ] Configure your search defaults: `search_term`, `location` (e.g. `"Tel Aviv, Israel"`), `site_name` (which boards to include), `is_remote`, `results_wanted`
- [ ] Add it to your MCP client config (or, if using LangChain instead of raw MCP client, wrap the same call in an `@tool`-decorated function so it fits the same tool-calling pattern as your other tools)
- [ ] Test standalone — run one search and confirm real postings come back with title, company, description, link, and date fields populated

**Notes carried over from the original plan:**
- You still control **pagination and volume from your side** — cap `results_wanted` per run (e.g. 20–50) rather than letting the server return everything it can scrape
- **Rate limits / blocking still apply** — since this is scraping under the hood, add basic retry/backoff and don't assume every board in `site_name` will succeed every run; treat partial results as normal, not a failure
- If a specific board (e.g. LinkedIn) gets blocked in testing, fall back to whichever boards are working rather than blocking the whole pipeline on one source

**Milestone:** you can pull real, live postings for Israel into a Python list via the JobSpy MCP tool.

---

## Phase 2.5 — Vector Store & Semantic Layer

Introduces the vector DB and the NLP fundamentals. Nothing LLM-facing yet — this all runs between JobSpy and the agent.

- [ ] Create `tools/vector_store.py` — a thin Chroma wrapper, persistent at `./vector_store/`, with two collections:
  - `cv_chunks` — bullet-level chunks of your CV and any past cover letters
  - `postings` — one entry per scraped posting, embedded from `title + company + description`
- [ ] Embeddings: `GoogleGenerativeAIEmbeddings` on `gemini-embedding-001` — reuses your existing Gemini key
- [ ] Write the CV chunker: split at bullet/role boundaries rather than fixed character counts, targeting roughly 50–150 tokens per chunk. Store `role`, `company`, `date_range` as metadata so retrieved bullets can be cited later. Expect 60–100 chunks from one CV.
- [ ] Index the CV once as a setup step, alongside the Phase 1 `profile.json` generation
- [ ] **Semantic pre-ranking** — embed the profile's target-role text, embed each posting from a JobSpy run, rank by cosine, keep the top 30
- [ ] **Near-duplicate detection** — flag posting pairs above a cosine threshold (start at 0.92, tune by eye on real data) as the same role cross-posted to different boards. This catches what the link-based dedup in Phase 6 cannot.
- [ ] Persist posting embeddings so reruns don't re-embed — free-tier embedding quota is finite

**Milestone:** a JobSpy run of 50 postings collapses to ~30 deduped, cosine-ranked candidates, and you can inspect the ranking to confirm it's sane before any LLM sees it.

---

## Phase 2.6 — Cross-Encoder Reranker

Phase 2.5's cosine ranking is a **bi-encoder**: profile and posting are embedded separately, then compared. Fast and precomputable, but the two texts never interact inside the model, so it misses constraint mismatches like "8+ years required" against a 2-year profile.

A **cross-encoder** feeds the pair through one transformer as `[profile] [SEP] [posting]`, with full cross-attention between them, and emits a single relevance score. Much more precise, but O(n) forward passes with nothing precomputable — so it only runs on a shortlist.

- [ ] Add `tools/reranker.py` using `sentence_transformers.CrossEncoder`
- [ ] Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` — ~80MB, 6 layers, scores 30 pairs on CPU in a couple of seconds. (`BAAI/bge-reranker-base` is stronger but ~1.1GB and noticeably slower; try it only after MiniLM works and you have Phase 8 numbers to compare against.)
- [ ] Build the query text from the profile — target roles, must-haves, key skills — not the raw CV dump
- [ ] Batch the pairs, and load the model once at module level rather than per call
- [ ] Wire the full cascade: **50 scraped → cosine top 30 → cross-encoder top 10 → Gemini**
- [ ] Log per-stage timing so the latency cost is a real number you can quote

**Milestone:** the cascade runs end to end, and you can print the cosine and cross-encoder rankings side by side and point at specific postings the reranker demoted for a reason cosine couldn't see.

---

## Phase 3 — Sheets Tool

- [ ] Create a Google Sheet + service account credentials (simpler than OAuth for a personal script)
- [ ] Use `gspread` to write a function that appends a row (company, title, match reasoning, status, link, date)
- [ ] Test standalone — confirm a real row appears in a real sheet

**Milestone:** running the function manually adds a row to the sheet.

---

## Phase 4 — Turn Functions into LangChain Tools

- [ ] Wrap each of your tool functions with LangChain's `@tool` decorator
- [ ] Write clear docstrings — LangChain generates the tool schema from the function signature and docstring, and this is what the LLM reads to decide when to use each tool
- [ ] Bind the tools to your Gemini model (`model.bind_tools([...])`)
- [ ] Note that `search_jobs` now returns the *cascade output* (top 10) rather than everything JobSpy scraped — the retrieval work from Phases 2.5–2.6 happens inside the tool, invisible to the agent

**Milestone:** `tool.invoke(...)` works for each tool individually, and the model can see all of them when queried.

---

## Phase 4.5 — RAG for Cover-Letter Drafting

Phases 2.5 and 2.6 are vector *search*, not RAG — nothing retrieved conditions a generation. This phase is what makes "RAG" accurate: retrieved text is injected into a prompt and shapes the output.

- [ ] Add a `draft_cover_letter` tool: given a posting, embed the JD, retrieve the top 5–8 `cv_chunks`, and inject only those into the Gemini prompt
- [ ] Prompt instruction: use only the supplied experience bullets; if the posting requires something not present in them, say so explicitly rather than inventing it
- [ ] Return the retrieved chunks alongside the letter, so every draft ships with its evidence — this is both the anti-hallucination mechanism and the demo visual
- [ ] Keep a no-RAG variant behind a flag (whole CV dumped into the prompt) purely so Phase 8 has something to compare against

**Milestone:** a generated letter displayed next to the exact CV bullets that produced it, with no claims that don't trace back to one of them.

---

## Phase 5 — Build the LangGraph Agent

This replaces a manual "call LLM → check for tool call → execute → loop" script with an explicit graph.

- [ ] Define a state object that holds the running list of messages/conversation
- [ ] Define an **agent node** (calls the LLM with current state)
- [ ] Define a **tools node** (executes whichever tool the LLM requested)
- [ ] Define a **conditional edge**: if the last LLM message includes a tool call → go to tools node; otherwise → end
- [ ] Look at LangGraph's prebuilt ReAct-agent pattern for reference before building the graph — but consider building it manually once yourself as a learning exercise

**Milestone:** invoking the graph with a goal causes it to autonomously loop through tool calls (search → evaluate → maybe log) until it produces a final answer, with no hardcoded call order.

---

## Phase 6 — Guardrails

- [ ] Pass a `recursion_limit` when invoking the graph (LangGraph's built-in max-step protection)
- [ ] Add a dedup check inside `log_posting`: skip postings whose link is already in the sheet (read the sheet back first)
- [ ] This is *exact-link* dedup and it stacks with the *semantic* dedup from Phase 2.5 — the link check catches the same posting logged twice across runs, the cosine check catches one role cross-posted to LinkedIn and Indeed under different URLs. You want both.
- [ ] Add a `DRY_RUN` env flag that logs what *would* be written without writing it

**Milestone:** running the graph twice on the same postings doesn't create duplicate rows, and a bad run can't loop forever.

---

## Phase 7 — Tracing / Demo Polish

- [ ] Use `.stream()` instead of `.invoke()` to watch the graph's node transitions and tool calls live
- [ ] Set up LangSmith (free tier) for a visual trace of every LLM reasoning step and tool call — good for screenshots in a portfolio writeup (see Phase 8.4; `langsmith` is already a dependency)
- [ ] (Optional) Wrap the entry point in `argparse` so you can run `python main.py --goal "find backend jobs in Tel Aviv"` from the CLI

**Milestone:** you can run one command, watch a readable trace of the agent's decisions scroll by, and use it as a talking point in interviews.

---

## Phase 8 — Evaluation

Evaluation is its own discipline — not a kind of fine-tuning, and not steering. There are two distinct layers here, and naming them separately matters. Everything in this phase runs **offline**, in an `eval/` directory, outside the agent loop.

Don't skip this. It's what turns "I added a reranker" into "reranking moved precision@10 from 0.4 to 0.7 at 1.8s added latency."

### 8.1 — Build the golden set

- [ ] Export ~40 real postings from JobSpy runs
- [ ] Label each yourself: 0 = bad match, 1 = plausible, 2 = strong match. Roughly an hour of work.
- [ ] Save as `eval/golden_set.json`
- [ ] **Do this early** — it gates 8.2, 8.3, Phase 9, and Phase 10, and it's the one step that can't be rushed at the end. The same 40 labels are also the training data for Phase 9: label once, use twice.

### 8.2 — IR / ranking evaluation

Classic information-retrieval metrics over the golden set.

- [ ] Implement precision@10, nDCG@10, and MRR in `eval/ranking_eval.py`
- [ ] Score four variants:
  - **V0** — BM25 keyword baseline (`rank-bm25`)
  - **V1** — bi-encoder cosine (Phase 2.5)
  - **V2** — cosine → cross-encoder rerank (Phase 2.6)
  - **V3** — LLM-only, Gemini scores all 40 directly
- [ ] Report wall-clock latency and token cost alongside each score

V3 may well win on raw quality. That's a fine outcome — the interesting finding is what it *costs* relative to V2, and whether the gap justifies it. That cost/quality table is the centerpiece result of the whole project.

### 8.3 — LLM output evaluation (groundedness)

- [ ] Generate 20 cover letters under both the RAG and no-RAG variants from Phase 4.5
- [ ] LLM-as-judge rubric: decompose each letter into factual claims, then mark each claim traceable / not traceable to a supplied CV chunk
- [ ] Metric: percentage of claims grounded, plus token count per letter
- [ ] Expected result: RAG matches or beats no-RAG on groundedness at a fraction of the prompt tokens. Either outcome is reportable.

### 8.4 — Tracing

- [ ] `langsmith` is already in `requirements.txt` — enable it for trace-level inspection of the agent loop (this is the Phase 7 optional item, now non-optional since it's cheap and produces good screenshots)

**Milestone:** one table quantifying the ranking cascade, one quantifying RAG groundedness.

---

## Phase 9 — Classical ML *(optional)*

Cheap, given the golden set from 8.1 already exists.

### 9.1 — Unsupervised (no labels needed)

- [ ] k-means and HDBSCAN over the posting embeddings from Phase 2.5
- [ ] Project to 2D with UMAP, plot with matplotlib, label clusters by inspecting their top terms
- [ ] Output: a scatter plot of which market segments the job boards actually surface to you

### 9.2 — Supervised (uses the 40 golden-set labels)

- [ ] Features: posting embedding, plus structured signals — remote flag, seniority parsed from title, source board, description length
- [ ] Model: logistic regression, then gradient boosting for comparison
- [ ] **k-fold cross-validation, and report the confidence interval.** With n=40 it will be wide.
- [ ] Compare against the Phase 2.6 cross-encoder on the same folds

Be explicit that n=40 is small. "CV precision 0.72 ± 0.11, and here's why the interval is wide" is a stronger result than one impressive number with no error bars. Correct methodology on small data is the point.

**Milestone:** a cluster plot and a cross-validated classifier with honest error bars.

---

## Phase 10 — Fine-Tuning *(optional — expect a negative result, and say so up front)*

Included to cover the topic honestly rather than to win.

- [ ] Fine-tune `ms-marco-MiniLM-L-6-v2` on the 40 golden-set labels
- [ ] k-fold, evaluated with the same 8.2 metrics so the numbers are directly comparable
- [ ] Predicted outcome: it **underperforms** the pretrained baseline, because 40 examples can't beat the millions of pairs MS MARCO was trained on
- [ ] The deliverable is the analysis — what data volume would actually be needed, what overfitting looks like in the fold-by-fold numbers, and why "we tried it and it lost" was the correct call

A documented negative result with sound methodology is a legitimate finding. A fabricated win is not.

---

## Key Design Principle to Keep in Mind

Nowhere in this build should there be a line like `if match_score > 0.7: log_posting(...)`. The LLM itself should decide, based on the profile and posting it's given, whether something is worth logging and drafting for — and explain why. That's what makes this an *agentic* app rather than a script with an LLM bolted on.

**This gets harder to honor once Phases 2.6 and 9 exist**, because then you have a cross-encoder score and a classifier probability sitting right there, and thresholding on them is faster and cheaper than calling Gemini. The line to hold:

- **Filtering is fine.** The cascade cuts 50 postings to 10 because quota and latency won't support reasoning over 50. The cut point is arbitrary — it'd be 15 with a bigger budget — no posting is declared a bad match, and nothing is written anywhere.
- **Deciding is not.** Auto-logging or auto-rejecting asserts "this is worth applying to" or "this isn't," and acts on it by writing a verdict to your sheet.

The tell is the sheet schema: it has a **match reasoning** column. A cross-encoder emits `0.34` and nothing else — it can't fill that column, because it has no reasons, only a score. If a threshold fills it with "score below cutoff," you've replaced an explanation with a restatement of the number.

There's an honesty dimension too: the Phase 9 model trains on 40 labels with a wide confidence interval. Using it to reorder a shortlist is well within what that supports. Using it to silently discard postings isn't.

**Short version: scores may change the order and the size of what Gemini looks at. Scores may not write to the sheet.**

---

## Suggested Build Order Recap

1. Phase 0–3: build and test each tool in isolation (plain Python, no LLM)
2. Phase 8.1: label the golden set — do it early, it gates four later phases
3. Phase 2.5: vector store, semantic dedup, cosine ranking
4. Phase 2.6: cross-encoder cascade
5. Phase 8.2: rank the four variants — your first real numbers
6. Phase 4: wrap tools for LangChain
7. Phase 4.5: RAG cover letters, then Phase 8.3 to measure their groundedness
8. Phase 5: build the LangGraph loop — start with just ONE tool wired in before adding the rest
9. Phase 6: guardrails
10. Phase 7 + 8.4: tracing and polish
11. Phases 9 and 10 as time allows

Phases 0–8 are the closing project; 9 and 10 are bonus slides.

Build it one phase at a time and resist the urge to jump to Phase 5 before Phases 1–3 are solid; debugging the agent loop is much harder if you're not sure your tools work correctly on their own first. The same logic applies to the retrieval work — get the cascade producing sane rankings on its own (Phase 2.6's milestone) before wiring it into a tool the agent calls.
