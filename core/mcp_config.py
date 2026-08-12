"""
mcp_config.py

description: Registry of MCP servers this agent host connects to. Add new servers here as additional dict entries - no other code changes needed for MultiServerMCPClient to pick them up.
"""

import os
from pathlib import Path

from integrations.sheets_auth import materialize_oauth_client_file
from core.logger import get_logger

logger = get_logger(__name__)

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
        "env": _stdio_env_override(ENABLE_SSE="0")
    },
    # Open-source Sheets server (github.com/xing5/mcp-google-sheets), run on demand
    # via uvx. Used instead of Google's official remote server
    # (https://sheetsmcp.googleapis.com/mcp/v1), which is gated behind the Workspace
    # Developer Preview Program - that one returns "The caller does not have
    # permission" on every call until access is granted, even with valid full-scope
    # credentials. Reuses the same OAuth client as the direct-API path in
    # tools/sheets_logging.py, so there's only one set of credentials to manage.
    "sheets": {
        "transport": "stdio",
        "command": "uvx",
        # `--with "mcp<2"` is required: mcp-google-sheets declares `mcp>=1.8.0`, so uv
        # otherwise resolves mcp 2.0.0, which removed `mcp.server.fastmcp` and crashes
        # the server on import. Drop the pin once upstream supports the 2.x SDK.
        "args": ["--with", "mcp<2", "mcp-google-sheets@latest"],
        "env": _stdio_env_override(
            CREDENTIALS_PATH=str(materialize_oauth_client_file()),
            TOKEN_PATH=str(_REPO_ROOT / ".google_tokens" / "mcp_sheets_token.json"),
        ),
    },
}

logger.info("Registered MCP servers: %s", list(MCP_SERVERS))