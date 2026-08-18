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

Phase 6 guardrails, both backends: exact-link dedup (skip a row whose link is
already in the sheet - stacks with the semantic near-dup check in
services/vector_store.py, which catches the same role cross-posted under a
different URL) and a DRY_RUN env flag (logs what would be written instead of
writing it). Both functions now return a status string - "appended",
"skipped_duplicate", or "dry_run" - instead of None.

Note on the MCP path: mcp-google-sheets has no "append" tool - `add_rows` only
inserts blank rows - so appending is read-then-write (get_sheet_data to find
the first empty row, update_cells to fill it). gspread's append_row() does
that server-side in one call, which is why the direct path is shorter.

usage: python -m integrations.sheets_logging          # direct API
       python -m integrations.sheets_logging --mcp    # via MCP server
"""
import json
import os
import sys

import gspread
from dotenv import load_dotenv
from rich.console import Console

from integrations.sheets_auth import get_sheets_credentials
from core.logger import get_logger

load_dotenv()
console = Console()
logger = get_logger(__name__)

_SHEET_TAB = "Sheet1"  # rename here if your tab is named differently
_ROW_FIELDS = ["company", "title", "match_reasoning", "status", "link", "date", "cv_profile_path"]


def _row_values(row: dict) -> list[str]:
    """Validate a row dict and flatten it into sheet column order."""
    missing = [field for field in _ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"row is missing required field(s): {missing}")
    return [str(row[field]) for field in _ROW_FIELDS]


def _resolve_sheet_id(spreadsheet_id: str | None) -> str:
    return spreadsheet_id or os.environ["GOOGLE_SHEET_ID"]


def _is_dry_run() -> bool:
    return os.environ.get("DRY_RUN", "").strip().lower() in ("1", "true", "yes")


_LINK_FIELD_INDEX = _ROW_FIELDS.index("link")


# --- direct Sheets API (gspread) ------------------------------------------

def _get_worksheet(spreadsheet_id: str | None = None):
    """Open the tracking sheet's worksheet, authorizing if needed."""
    client = gspread.authorize(get_sheets_credentials())
    return client.open_by_key(_resolve_sheet_id(spreadsheet_id)).worksheet(_SHEET_TAB)


def _get_logged_links(worksheet) -> set[str]:
    """Every link already in the sheet, read from the same worksheet object
    the append will use, so no extra round trip through Sheets auth."""
    rows = worksheet.get_all_values()[1:]  # skip header
    return {row[_LINK_FIELD_INDEX] for row in rows if len(row) > _LINK_FIELD_INDEX}


def log_job_match(row: dict, spreadsheet_id: str | None = None) -> str:
    """Append one matched-job row to the tracking sheet via the Sheets API.

    Guardrails: skips rows whose link is already logged (exact-link dedup),
    and honors DRY_RUN (logs what would be written, writes nothing).

    Args:
        row (dict): must contain all of _ROW_FIELDS
            (company, title, match_reasoning, status, link, date).
        spreadsheet_id (str | None): defaults to GOOGLE_SHEET_ID from .env.

    Returns:
        str: "appended", "skipped_duplicate", or "dry_run".
    """
    values = _row_values(row)
    worksheet = _get_worksheet(spreadsheet_id)

    if row["link"] in _get_logged_links(worksheet):
        logger.info("Skipping duplicate link for %r: %s", row.get("company"), row["link"])
        return "skipped_duplicate"

    if _is_dry_run():
        logger.info("[DRY_RUN] would append row for %r: %s", row.get("company"), row)
        return "dry_run"

    logger.info("Appending job match row for %r to Sheets API", row.get("company"))
    try:
        worksheet.append_row(values, value_input_option="USER_ENTERED")
    except Exception:
        logger.exception("Failed to append row via Sheets API")
        raise
    logger.info("Row appended via Sheets API")
    return "appended"


# --- via the mcp-google-sheets MCP server ---------------------------------

async def get_sheets_tools():
    """Return LangChain tools scoped to sheets (the mcp-google-sheets server).

    Returns:
        list[BaseTool]: e.g. `sheets_get_sheet_data`, `sheets_update_cells`,
        ready to bind to a LangGraph/LangChain agent.
    """
    from core.mcp_client import get_mcp_tools

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


async def log_job_match_via_mcp(row: dict, spreadsheet_id: str | None = None) -> str:
    """Append one matched-job row to the tracking sheet through the MCP server.

    Guardrails: skips rows whose link is already logged (exact-link dedup),
    and honors DRY_RUN (logs what would be written, writes nothing).

    Args:
        row (dict): must contain all of _ROW_FIELDS.
        spreadsheet_id (str | None): defaults to GOOGLE_SHEET_ID from .env.

    Returns:
        str: "appended", "skipped_duplicate", or "dry_run".
    """
    values = _row_values(row)
    spreadsheet_id = _resolve_sheet_id(spreadsheet_id)
    try:
        tools = {tool.name: tool for tool in await get_sheets_tools()}

        # A:G covers every _ROW_FIELDS column - one read gives both the
        # existing links (for dedup) and the row count (for the next append).
        existing = _parse_mcp_result(await tools["sheets_get_sheet_data"].ainvoke(
            {"spreadsheet_id": spreadsheet_id, "sheet": _SHEET_TAB, "range": "A:G"}
        ))
        # shape: {"spreadsheetId": ..., "valueRanges": [{"range": ..., "values": [[...], ...]}]}
        value_ranges = existing.get("valueRanges") or [{}]
        existing_rows = (value_ranges[0].get("values") or [])[1:]  # skip header
        existing_links = {r[_LINK_FIELD_INDEX] for r in existing_rows if len(r) > _LINK_FIELD_INDEX}
        next_row = len(existing_rows) + 2  # +1 for header, +1 for 1-indexing

        if row["link"] in existing_links:
            logger.info("Skipping duplicate link for %r: %s", row.get("company"), row["link"])
            return "skipped_duplicate"

        if _is_dry_run():
            logger.info("[DRY_RUN] would append row for %r via MCP: %s", row.get("company"), row)
            return "dry_run"

        logger.info("Appending job match row for %r via MCP sheets server", row.get("company"))
        await tools["sheets_update_cells"].ainvoke(
            {
                "spreadsheet_id": spreadsheet_id,
                "sheet": _SHEET_TAB,
                "range": f"A{next_row}:G{next_row}",
                "data": [values],
            }
        )
    except Exception:
        logger.exception("Failed to append row via MCP sheets server")
        raise
    logger.info("Row appended via MCP sheets server at row %d", next_row)
    return "appended"


# --- smoke test -----------------------------------------------------------

_TEST_ROW = {
    "company": "Test Co",
    "title": "Test Role",
    "match_reasoning": "smoke test row",
    "status": "n/a",
    "link": "https://example.com",
    "date": "2026-08-09",
    "cv_profile_path": "",
}


def _smoke_test(use_mcp: bool, spreadsheet_id: str | None = None):
    # _TEST_ROW's link is fixed, so running this twice in a row demonstrates
    # the Phase 6 dedup guardrail: 1st run -> "appended", 2nd -> "skipped_duplicate".
    if use_mcp:
        import asyncio
        status = asyncio.run(log_job_match_via_mcp(_TEST_ROW, spreadsheet_id))
        console.print(f"[bold green]MCP path status: {status}[/bold green]")
    else:
        status = log_job_match(_TEST_ROW, spreadsheet_id)
        console.print(f"[bold green]Sheets API status: {status}[/bold green]")

    console.print("[bold]Sheet now contains:[/bold]")
    console.print(_get_worksheet(spreadsheet_id).get_all_values())


if __name__ == "__main__":
    _smoke_test(use_mcp="--mcp" in sys.argv)
