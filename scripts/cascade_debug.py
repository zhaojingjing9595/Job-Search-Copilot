"""
cascade_debug.py

description: Dev-only visualization of the retrieval cascade (Phases 2.5/2.6).
Cosine-ranks a batch, then cross-encoder-scores *every* candidate in that
shortlist without truncating early, and logs each one to a separate
"CascadeDebug" tab in the tracking spreadsheet - cosine (bi-encoder) score,
cross-encoder score, and whether it survived the final cut - so the cascade's
filtering behavior is inspectable and visualizable in one place, run over run,
instead of re-derived from console logs each time.

This never touches the tracking sheet's main tab or its row schema (see
integrations/sheets_logging.py) - it's a parallel, disposable debug trail
that reuses the same spreadsheet (GOOGLE_SHEET_ID) but its own tab.

usage: python -m scripts.cascade_debug            # cache-or-scrape, log + print
       python -m scripts.cascade_debug --refresh  # force a fresh scrape first
"""
import asyncio
import os
import sys
from datetime import datetime, timezone

import gspread
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from core.logger import get_logger
from integrations.sheets_auth import get_sheets_credentials
from services.reranker import rerank_postings
from services.vector_store import index_postings, load_profile, rank_postings_by_profile

load_dotenv()
console = Console()
logger = get_logger(__name__)

_DEBUG_TAB = "CascadeDebug"
_HEADER = ["run_id", "title", "company", "description", "cosine_similarity", "cosine_rank",
           "cross_encoder_rank", "rerank_score", "status"]


def _get_clean_debug_worksheet(spreadsheet_id: str, rows: int = 2000):
    """Get the CascadeDebug tab, wiped to just the header row - each run
    replaces the previous run's rows rather than accumulating alongside them.
    """
    client = gspread.authorize(get_sheets_credentials())
    spreadsheet = client.open_by_key(spreadsheet_id)
    try:
        worksheet = spreadsheet.worksheet(_DEBUG_TAB)
        worksheet.clear()
        logger.info("Cleared existing %r tab", _DEBUG_TAB)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=_DEBUG_TAB, rows=rows, cols=len(_HEADER))
        logger.info("Created %r tab in spreadsheet %s", _DEBUG_TAB, spreadsheet_id)
    worksheet.append_row(_HEADER)
    return worksheet


def score_cascade(
    profile: dict,
    cosine_top_k: int = 30,
    cross_encoder_top_k: int = 10,
    query_vector: list[float] | None = None,
) -> list[dict]:
    """Cosine-rank, then cross-encoder-score every cosine candidate - not just
    the ones that would survive the real cascade's cutoff - so the full
    shortlist is inspectable, not only the winners.

    Args:
        query_vector: precomputed profile-query embedding (e.g. from
            scripts.dev_cache.cached_profile_query_embedding) to skip the
            live embed call. Defaults to None - embeds live.

    Returns:
        list[dict]: cosine_top_k candidates (empty if nothing indexed yet),
        each with cosine_rank, cross_encoder_rank, rerank_score, and status
        ("selected" if within cross_encoder_top_k else "filtered_out"),
        sorted by cross-encoder rank (best first).
    """
    cosine_ranked = rank_postings_by_profile(profile, top_k=cosine_top_k, query_vector=query_vector)
    if not cosine_ranked:
        return []
    for i, c in enumerate(cosine_ranked, start=1):
        c["cosine_rank"] = i

    # top_k = len(...): nothing gets truncated here, unlike the real cascade -
    # we want a score for every candidate, not just the ones that would win.
    scored = rerank_postings(profile, cosine_ranked, top_k=len(cosine_ranked))
    for i, c in enumerate(scored, start=1):
        c["cross_encoder_rank"] = i
        c["status"] = "selected" if i <= cross_encoder_top_k else "filtered_out"
    return scored


def log_cascade_debug(scored: list[dict], spreadsheet_id: str, run_id: str | None = None) -> str:
    """Clear the CascadeDebug tab and write every scored candidate as one
    row - this run's results replace whatever was there before.

    Returns:
        str: the run_id all rows in this batch were tagged with.
    """
    run_id = run_id or datetime.now(timezone.utc).isoformat(timespec="seconds")
    worksheet = _get_clean_debug_worksheet(spreadsheet_id)
    rows = [
        [
            run_id,
            c["metadata"].get("title"),
            c["metadata"].get("company"),
            c["metadata"].get("description"),
            round(c["similarity"], 4),
            c["cosine_rank"],
            c["cross_encoder_rank"],
            round(c["rerank_score"], 4),
            c["status"],
        ]
        for c in scored
    ]
    worksheet.append_rows(rows, value_input_option="USER_ENTERED")
    logger.info("Logged %d cascade debug row(s) under run_id=%s", len(rows), run_id)
    return run_id


def _print_table(scored: list[dict]) -> None:
    table = Table(title="Cosine (bi-encoder) -> cross-encoder cascade")
    for col in ["Cosine rank", "Cosine sim", "Cross-enc rank", "Rerank score", "Status", "Title", "Company"]:
        table.add_column(col)
    for c in scored:
        style = "green" if c["status"] == "selected" else "dim"
        table.add_row(
            str(c["cosine_rank"]), f"{c['similarity']:.3f}",
            str(c["cross_encoder_rank"]), f"{c['rerank_score']:.3f}",
            c["status"], str(c["metadata"].get("title")), str(c["metadata"].get("company")),
            style=style,
        )
    console.print(table)


async def _main():
    from scripts.dev_cache import cached_profile_query_embedding, cached_search_jobs

    force_refresh = "--refresh" in sys.argv
    profile = load_profile()
    query_vector = cached_profile_query_embedding(profile, force_refresh=force_refresh)

    postings = await cached_search_jobs(
        force_refresh=force_refresh,
        searchTerm="Full Stack Developer", location="Tel Aviv",
        countryIndeed="Israel", resultsWanted=50, siteNames="indeed,linkedin",
        hoursOld=168, format="json",
    )
    index_postings(postings)

    scored = score_cascade(profile, query_vector=query_vector)
    if not scored:
        console.print("[yellow]No postings in the vector store - nothing to score.[/yellow]")
        return
    _print_table(scored)

    run_id = log_cascade_debug(scored, os.environ["GOOGLE_SHEET_ID"])
    console.print(f"[green]Logged {len(scored)} row(s) to the {_DEBUG_TAB!r} tab under run_id={run_id}.[/green]")


if __name__ == "__main__":
    asyncio.run(_main())
