"""
jobs_searching.py

description: Job-search-specific tool loading - filters the shared MCP
client (tools/mcp_client.py) down to tools exposed by the jobspy server,
which wraps a Dockerized jobspy scraper behind a `search_jobs` tool.

usage: python -m tools.jobs_searching   # standalone smoke test
"""
from rich.console import Console

from tools.mcp_client import get_mcp_tools

console = Console()

async def get_job_search_tools():
    """Return LangChain tools scoped to job search (the jobspy server).

    Returns:
        list[BaseTool]: e.g. `jobspy_search_jobs`, ready to bind to a
        LangGraph/LangChain agent.
    """
    tools = await get_mcp_tools()
    return [tool for tool in tools if tool.name.startswith("jobspy_")]

async def _smoke_test():
    tools = await get_job_search_tools()
    console.print(f"[green]Loaded {len(tools)} MCP tool(s):[/green] "
                  f"{[t.name for t in tools]}")
    if tools:
        result = await tools[0].ainvoke(
            {
                "searchTerm": "Full Stack engineer", 
                "location": "Tel Aviv", 
                "countryIndeed": "Israel", 
                "resultsWanted": 5,
                "format": "json"
            }
        )
        console.print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke_test())
