"""
jobs_searching.py

description: Job-search-specific tool loading - filters the shared MCP
client (tools/mcp_client.py) down to tools exposed by the jobspy server,
which wraps a Dockerized jobspy scraper behind a `search_jobs` tool.

usage: python -m integrations.jobs_searching   # standalone smoke test
"""
import json
import re

from rich.console import Console

from core.mcp_client import get_mcp_tools
from core.logger import get_logger

console = Console()
logger = get_logger(__name__)

_HEBREW_RE = re.compile(r"[֐-׿]")


def _is_hebrew(posting: dict) -> bool:
    text = f"{posting.get('title', '')} {posting.get('description', '')}"
    return bool(_HEBREW_RE.search(text))

async def get_job_search_tools():
    """Return LangChain tools scoped to job search (the jobspy server).

    Returns:
        list[BaseTool]: e.g. `jobspy_search_jobs`, ready to bind to a
        LangGraph/LangChain agent.
    """
    tools = await get_mcp_tools()
    job_tools = [tool for tool in tools if tool.name.startswith("jobspy_")]
    logger.info("Scoped %d job-search tool(s): %s", len(job_tools), [t.name for t in job_tools])
    return job_tools


def extract_postings(raw) -> list[dict]:
    """Unwrap a jobspy_search_jobs tool response into its list of postings.

    ainvoke() returns MCP content blocks - a list of {"type": "text", "text":
    "<json>"} dicts - rather than a parsed dict, so the JSON payload has to be
    pulled out of the first text block before it can be read.
    """
    if isinstance(raw, list):
        raw = raw[0]["text"]
    payload = json.loads(raw) if isinstance(raw, str) else raw
    return payload.get("jobs", [])


async def search_jobs(**params) -> list[dict]:
    """Run jobspy_search_jobs standalone and return its postings.

    Args:
        **params: passed straight through to the tool (searchTerm, location,
            countryIndeed, resultsWanted, format, ...).

    Returns:
        list[dict]: postings as returned by JobSpy (title, company, skills,
        description, location, ...).
    """
    tools = await get_job_search_tools()
    search_tool = next((t for t in tools if t.name == "jobspy_search_jobs"), None)
    if search_tool is None:
        raise RuntimeError("jobspy_search_jobs tool not found - is the MCP server configured?")

    logger.info("Invoking %s with %s", search_tool.name, params)
    try:
        raw = await search_tool.ainvoke(params)
    except Exception:
        logger.exception("%s invocation failed", search_tool.name)
        raise
    postings = extract_postings(raw)
    logger.info("%s succeeded - %d posting(s)", search_tool.name, len(postings))
    return postings


async def _smoke_test():
    postings = await search_jobs(
        searchTerm="Full Stack Developer",
        location="Tel Aviv",
        countryIndeed="Israel",
        resultsWanted=50,
        # glassdoor deliberately excluded: JobSpy's own library raises
        # "Glassdoor is not available for ISRAEL" and crashes the whole
        # scrape rather than skipping just that board
        siteNames="indeed,linkedin",
        hoursOld=168,
        format="json",
    )

    logger.info("Full posting payload:\n%s", json.dumps(postings, indent=2))
    console.print_json(json.dumps(postings))

if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke_test())
