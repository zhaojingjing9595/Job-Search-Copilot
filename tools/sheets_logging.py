"""
sheets_logging.py

description: Appends matched job postings to the tracking Google Sheet. Two
interchangeable backends, kept side by side on purpose:

  log_job_match()          - sync, talks straight to the Sheets API via gspread
  log_job_match_via_mcp()  - async, goes through the mcp-google-sheets server
                             registered in mcp_config.py

Same arguments, same resulting row. The MCP path is the one to study for how
tool calls flow through an MCP server; the direct path is the dependency-free
fallback (no subprocess, no uvx) if the server misbehaves.

Note on the MCP path: mcp-google-sheets has no "append" tool - `add_rows` only
inserts blank rows - so appending is read-then-write (get_sheet_data to find
the first empty row, update_cells to fill it). gspread's append_row() does
that server-side in one call, which is why the direct path is shorter.

usage: python -m tools.sheets_logging          # direct API
       python -m tools.sheets_logging --mcp    # via MCP server
"""
import json
import os
import sys

import gspread
from dotenv import load_dotenv
from rich.console import Console

from tools.sheets_auth import get_sheets_credentials

load_dotenv()
console = Console()

_SHEET_TAB = "Sheet1"  # rename here if your tab is named differently
_ROW_FIELDS = ["company", "title", "match_reasoning", "status", "link", "date"]


def _row_values(row: dict) -> list[str]:
    """Validate a row dict and flatten it into sheet column order."""
    missing = [field for field in _ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"row is missing required field(s): {missing}")
    return [str(row[field]) for field in _ROW_FIELDS]


def _resolve_sheet_id(spreadsheet_id: str | None) -> str:
    return spreadsheet_id or os.environ["GOOGLE_SHEET_ID"]


# --- direct Sheets API (gspread) ------------------------------------------

def _get_worksheet(spreadsheet_id: str | None = None):
    """Open the tracking sheet's worksheet, authorizing if needed."""
    client = gspread.authorize(get_sheets_credentials())
    return client.open_by_key(_resolve_sheet_id(spreadsheet_id)).worksheet(_SHEET_TAB)


def log_job_match(row: dict, spreadsheet_id: str | None = None) -> None:
    """Append one matched-job row to the tracking sheet via the Sheets API.

    Args:
        row (dict): must contain all of _ROW_FIELDS
            (company, title, match_reasoning, status, link, date).
        spreadsheet_id (str | None): defaults to GOOGLE_SHEET_ID from .env.
    """
    _get_worksheet(spreadsheet_id).append_row(
        _row_values(row), value_input_option="USER_ENTERED"
    )


# --- via the mcp-google-sheets MCP server ---------------------------------

async def get_sheets_tools():
    """Return LangChain tools scoped to sheets (the mcp-google-sheets server).

    Returns:
        list[BaseTool]: e.g. `sheets_get_sheet_data`, `sheets_update_cells`,
        ready to bind to a LangGraph/LangChain agent.
    """
    from tools.mcp_client import get_mcp_tools

    tools = await get_mcp_tools()
    return [tool for tool in tools if tool.name.startswith("sheets_")]


def _parse_mcp_result(result):
    """MCP tools return content blocks; pull the JSON payload out of the first
    text block so callers get a plain dict."""
    if isinstance(result, list) and result and isinstance(result[0], dict):
        text = result[0].get("text", "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
    return result


async def log_job_match_via_mcp(row: dict, spreadsheet_id: str | None = None) -> None:
    """Append one matched-job row to the tracking sheet through the MCP server.

    Args:
        row (dict): must contain all of _ROW_FIELDS.
        spreadsheet_id (str | None): defaults to GOOGLE_SHEET_ID from .env.
    """
    values = _row_values(row)
    spreadsheet_id = _resolve_sheet_id(spreadsheet_id)
    tools = {tool.name: tool for tool in await get_sheets_tools()}

    existing = _parse_mcp_result(await tools["sheets_get_sheet_data"].ainvoke(
        {"spreadsheet_id": spreadsheet_id, "sheet": _SHEET_TAB, "range": "A:A"}
    ))
    # shape: {"spreadsheetId": ..., "valueRanges": [{"range": ..., "values": [[...], ...]}]}
    value_ranges = existing.get("valueRanges") or [{}]
    next_row = len(value_ranges[0].get("values") or []) + 1

    await tools["sheets_update_cells"].ainvoke(
        {
            "spreadsheet_id": spreadsheet_id,
            "sheet": _SHEET_TAB,
            "range": f"A{next_row}:F{next_row}",
            "data": [values],
        }
    )


# --- smoke test -----------------------------------------------------------

_TEST_ROW = {
    "company": "Test Co",
    "title": "Test Role",
    "match_reasoning": "smoke test row",
    "status": "n/a",
    "link": "https://example.com",
    "date": "2026-08-09",
}


def _smoke_test(use_mcp: bool, spreadsheet_id: str | None = None):
    if use_mcp:
        import asyncio
        asyncio.run(log_job_match_via_mcp(_TEST_ROW, spreadsheet_id))
        console.print("[bold green]Row appended via MCP server.[/bold green]")
    else:
        log_job_match(_TEST_ROW, spreadsheet_id)
        console.print("[bold green]Row appended via Sheets API.[/bold green]")

    console.print("[bold]Sheet now contains:[/bold]")
    console.print(_get_worksheet(spreadsheet_id).get_all_values())


if __name__ == "__main__":
    _smoke_test(use_mcp="--mcp" in sys.argv)
