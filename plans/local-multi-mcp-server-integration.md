# Local multi-MCP-server integration for Job-Search-Copilot

## Context

The user's agent (`Job-Search-Copilot`) is a Python/LangChain/langgraph project (Gemini via `langchain-google-genai`). They've already vendored a third-party Node.js MCP server (`job_spy_mcp/jobspy-mcp-server/`, cloned from `borgius/jobspy-mcp-server`) that wraps the `jobspy` scraping library behind an MCP `search_jobs` tool. This server **replaces** the previously-planned Adzuna integration — `ADZUNA_API_KEY`/`ADZUNA_APP_ID` in `.env` are leftover, not a second server to build toward. The user's explicit goal: their own Python agent should act as the MCP **host**, architected to support connecting to multiple MCP servers over time (the registry pattern below stays useful even with just one server today). This pass wires up that host-side capability and gets `search_jobs` callable end-to-end — it does not build the agent loop itself (`main.py` is still a scratch file with no established tool-binding pattern; that's separate follow-up work).

## Prerequisites (manual, before/alongside implementation)

1. **Start Docker Desktop** — checked live, `docker info` currently fails (daemon not running). Needed because `search_jobs` shells out to `docker run --rm jobspy ...`.
2. **Fix image tag mismatch.** User already built an image, but tagged it `jobspy-mcp-server`. The Node code hardcodes the image name — `src/tools/search-jobs.js` runs `docker run --rm jobspy ...` (the `JOBSPY_DOCKER_IMAGE` env var mentioned in the README is not actually referenced anywhere in `src/`, so it has no effect). Fix by retagging the existing image rather than rebuilding:
   ```
   docker tag jobspy-mcp-server jobspy
   ```
   Verify with `docker images | grep jobspy` (should show a `jobspy:latest` entry).
3. Git tracking of `job_spy_mcp/` is left as-is per user decision — not touched by this task.

## Implementation

### 1. `requirements.txt` — add dependency
Add `langchain-mcp-adapters==0.3.0` (verified on PyPI: pins `mcp>=1.9.2`, `langchain-core<2.0.0,>=1.0.0` — compatible with the existing `langchain-core==1.5.1`/`langgraph==1.2.9`). Install into `.venv`.

### 2. `tools/mcp_config.py` (new) — extensible server registry
Pure config module, no I/O — a `MCP_SERVERS` dict keyed by server name, so adding a future server (e.g. Adzuna) is a one-entry addition, no other code changes.

Key detail verified against `mcp` SDK source: `StdioServerParameters.env`, if supplied, **replaces** the child process env entirely rather than merging with the parent's — it does NOT inherit `PATH` automatically. Since `search-jobs.js` shells out to `docker`, the spawned Node process needs `PATH` forwarded or the `docker run` call inside it will fail with "command not found". Use a full copy of `os.environ` with `ENABLE_SSE` overridden to `"0"` (forces stdio transport regardless of what's in the third-party `.env`, without editing that file):

```python
"""
mcp_config.py

description: Registry of MCP servers this agent host connects to.
Add new servers here as additional dict entries — no other code changes
needed for MultiServerMCPClient to pick them up.
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_JOBSPY_SERVER_ENTRY = _REPO_ROOT / "job_spy_mcp" / "jobspy-mcp-server" / "src" / "index.js"

def _stdio_env_override(**overrides: str) -> dict:
    """Full copy of current process env with keys overridden.
    MCP's stdio client does NOT merge a supplied `env` dict with the parent
    process env — an explicit dict replaces it entirely, so PATH must be
    forwarded or anything the child shells out to (e.g. `docker`) won't resolve.
    """
    return {**os.environ, **overrides}

MCP_SERVERS = {
    "jobspy": {
        "transport": "stdio",
        "command": "node",
        "args": [str(_JOBSPY_SERVER_ENTRY)],
        "env": _stdio_env_override(ENABLE_SSE="0"),
    },
    # Future: "adzuna": {"transport": "stdio", "command": ..., "args": [...], "env": ...},
}
```

### 3. `tools/jobs_searching.py` (currently empty) — client + tool exposure
Follows `tools/profile.py`'s existing conventions (module docstring with description/usage, `rich.console.Console`, plain functions, `load_dotenv()`).

```python
"""
jobs_searching.py

description: Loads LangChain-compatible tools from all configured MCP
servers (see mcp_config.py) for the agent to call at runtime. Currently
wraps the jobspy-mcp-server (local Node subprocess, stdio transport) which
exposes a `search_jobs` tool backed by a Dockerized jobspy scraper.

usage: python -m tools.jobs_searching   # standalone smoke test
"""
from langchain_mcp_adapters.client import MultiServerMCPClient
from rich.console import Console
from dotenv import load_dotenv

from tools.mcp_config import MCP_SERVERS

load_dotenv()
console = Console()

_client = MultiServerMCPClient(MCP_SERVERS, tool_name_prefix=True)

async def get_job_search_tools():
    """Load and return LangChain tools from all configured MCP servers.

    Returns:
        list[BaseTool]: e.g. `jobspy_search_jobs`, ready to bind to a
        LangGraph/LangChain agent.
    """
    return await _client.get_tools()

async def _smoke_test():
    tools = await get_job_search_tools()
    console.print(f"[green]Loaded {len(tools)} MCP tool(s):[/green] "
                  f"{[t.name for t in tools]}")
    if tools:
        result = await tools[0].ainvoke(
            {"searchTerm": "software engineer", "location": "Remote", "resultsWanted": 5}
        )
        console.print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke_test())
```

Notes:
- `tool_name_prefix=True` on the client — prevents a future second server's tool names from colliding with jobspy's.
- Param names in the smoke test (`searchTerm`, `location`, `resultsWanted`) match the real camelCase schema in `job_spy_mcp/jobspy-mcp-server/src/schemas/searchParamsSchema.js` / `src/tools/search-jobs.js`, verified against source.
- `_client` instantiated once at import — cheap, no connection until `get_tools()` is awaited (per `MultiServerMCPClient`'s documented usage pattern).

## Explicitly out of scope

- `main.py` agent loop / LangGraph graph construction — separate follow-up once MCP tool loading is verified working.
- The `gemini-3.5-flash` model id in `constants.py`/`main.py` (doesn't appear to be a real shipped Gemini model id) — pre-existing, unrelated issue, flagged not fixed here.
- Adzuna — not in scope; jobspy-mcp-server replaces that plan, no second server to build toward right now.
- `job_spy_mcp/` git tracking strategy — left as-is per user decision.

## Verification

1. Confirm Docker Desktop running (`docker info`) and image tagged correctly (`docker images | grep jobspy` shows `jobspy:latest`, per the retag step above).
2. `pip install -r requirements.txt` in the `.venv`.
3. Run `python -m tools.jobs_searching` from repo root. Expected: spawns `node .../src/index.js` over stdio with `ENABLE_SSE=0`, lists `jobspy_search_jobs`, then actually invokes it — this makes a real network call to job boards via the Dockerized scraper and should print a handful of job results as JSON/rich output.
4. If it fails: check `docker info` first (most likely failure point), then check that `node` resolves inside the subprocess env (the `PATH`-forwarding fix in `mcp_config.py`).
