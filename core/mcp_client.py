"""
mcp_client.py

description: Central MultiServerMCPClient shared by every module that needs
MCP tools, built from every server registered in mcp_config.py. Add a server
to MCP_SERVERS there and its tools become available here automatically - no
other code changes needed.
"""
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

from core.mcp_config import MCP_SERVERS
from core.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_client = MultiServerMCPClient(MCP_SERVERS, tool_name_prefix=True)

async def get_mcp_tools():
    """Load and return LangChain tools from every configured MCP server.

    Returns:
        list[BaseTool]: all tools across all servers in MCP_SERVERS, each
        prefixed with its server's registry key (e.g. `jobspy_search_jobs`).
    """
    logger.info("Loading MCP tools from servers: %s", list(MCP_SERVERS))
    try:
        tools = await _client.get_tools()
    except Exception:
        logger.exception("Failed to load MCP tools")
        raise
    logger.info("Loaded %d MCP tool(s): %s", len(tools), [t.name for t in tools])
    return tools
