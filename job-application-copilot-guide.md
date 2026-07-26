# Job Application Copilot — Build Guide

An agentic app that searches job postings, evaluates them against your profile, drafts cover letters for good matches, and logs everything to a tracking sheet — with an LLM (Gemini) deciding what to do next at each step, not a hardcoded pipeline.

**Stack:** Python, LangChain, LangGraph, Gemini (free tier)

---

## Architecture Overview

```
Goal (e.g. "find backend jobs in Tel Aviv and evaluate matches")
        │
        ▼
   LangGraph Agent Loop
        │
   ┌────┴─────┬──────────────┬───────────────┐
   ▼           ▼              ▼               ▼
search_jobs  get_profile   log_posting    (LLM reasoning
  tool         tool           tool         between calls)
```

The LLM decides which tool to call, when to call it, whether a posting is a match, and when it's done — that decision-making is the "agentic" part.

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
  └── tools/
      ├── jobs.py
      ├── profile.py
      └── sheets.py
  ```
- [ ] Install core dependencies:
  ```
  pip install langchain langchain-google-genai langgraph python-dotenv gspread requests
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

## Phase 2 — Job Board Tool

- [ ] Write a function that calls your job board API given a query/location and returns a list of postings (title, company, description, link, date)
- [ ] Handle basic pagination/rate limits
- [ ] Test standalone — print real results to console

**Milestone:** you can pull real, live postings into a Python list.

---

## Phase 3 — Sheets Tool

- [ ] Create a Google Sheet + service account credentials (simpler than OAuth for a personal script)
- [ ] Use `gspread` to write a function that appends a row (company, title, match reasoning, status, link, date)
- [ ] Test standalone — confirm a real row appears in a real sheet

**Milestone:** running the function manually adds a row to the sheet.

---

## Phase 4 — Turn Functions into LangChain Tools

- [ ] Wrap each of your three functions with LangChain's `@tool` decorator
- [ ] Write clear docstrings — LangChain generates the tool schema from the function signature and docstring, and this is what the LLM reads to decide when to use each tool
- [ ] Bind the tools to your Gemini model (`model.bind_tools([...])`)

**Milestone:** `tool.invoke(...)` works for each tool individually, and the model can see all three when queried.

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
- [ ] Add a `DRY_RUN` env flag that logs what *would* be written without writing it

**Milestone:** running the graph twice on the same postings doesn't create duplicate rows, and a bad run can't loop forever.

---

## Phase 7 — Tracing / Demo Polish

- [ ] Use `.stream()` instead of `.invoke()` to watch the graph's node transitions and tool calls live
- [ ] (Optional) Set up LangSmith (free tier) for a visual trace of every LLM reasoning step and tool call — good for screenshots in a portfolio writeup
- [ ] (Optional) Wrap the entry point in `argparse` so you can run `python main.py --goal "find backend jobs in Tel Aviv"` from the CLI

**Milestone:** you can run one command, watch a readable trace of the agent's decisions scroll by, and use it as a talking point in interviews.

---

## Key Design Principle to Keep in Mind

Nowhere in this build should there be a line like `if match_score > 0.7: log_posting(...)`. The LLM itself should decide, based on the profile and posting it's given, whether something is worth logging and drafting for — and explain why. That's what makes this an *agentic* app rather than a script with an LLM bolted on.

---

## Suggested Build Order Recap

1. Phase 0–3: build and test each tool in isolation (plain Python, no LLM)
2. Phase 4: wrap tools for LangChain
3. Phase 5: build the LangGraph loop — start with just ONE tool wired in before adding the rest
4. Phase 6: guardrails
5. Phase 7: tracing and polish

Good luck — build it one phase at a time and resist the urge to jump to Phase 5 before Phases 1–3 are solid; debugging the agent loop is much harder if you're not sure your tools work correctly on their own first.
