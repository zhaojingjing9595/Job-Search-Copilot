"""
mcp_client.py

description: Central MultiServerMCPClient shared by every module that needs
MCP tools, built from every server registered in mcp_config.py. Add a server
to MCP_SERVERS there and its tools become available here automatically - no
other code changes needed.
"""
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

from tools.mcp_config import MCP_SERVERS

load_dotenv()

_client = MultiServerMCPClient(MCP_SERVERS, tool_name_prefix=True)

async def get_mcp_tools():
    """Load and return LangChain tools from every configured MCP server.

    Returns:
        list[BaseTool]: all tools across all servers in MCP_SERVERS, each
        prefixed with its server's registry key (e.g. `jobspy_search_jobs`).
    """
    return await _client.get_tools()
