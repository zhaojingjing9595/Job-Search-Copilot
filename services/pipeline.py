"""
pipeline.py

description: Orchestrates the full retrieval cascade - JobSpy scrape ->
Chroma index -> cosine rank (Phase 2.5) -> cross-encoder rerank (Phase
2.6). This is what Phase 4 will wrap as the `search_jobs` tool body.

Two modes, controlled by PIPELINE_MODE (env var, default "live") or the
`use_cache` argument (overrides the env var when passed explicitly):

  live  - calls the JobSpy MCP server and the Gemini embedding model, same
          as production. Costs one scrape + one embed call per run.
  cache - skips both. Reads whatever postings are already persisted in the
          `postings` Chroma collection (from a prior live run) and runs
          only the free part of the cascade (cosine rank + cross-encoder
          rerank, both local/precomputed). Use this while tuning the
          cascade itself so iteration doesn't burn JobSpy/Gemini quota.

usage: python -m services.pipeline                       # live
       PIPELINE_MODE=cache python -m services.pipeline    # cache
"""
import asyncio
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from core.logger import get_logger
from integrations.jobs_searching import search_jobs
from services.reranker import rerank_postings
from services.vector_store import (
    POSTINGS_COLLECTION,
    _get_collection,
    index_postings,
    load_profile,
    rank_postings_by_profile,
)

load_dotenv()
logger = get_logger(__name__)
console = Console()

CACHE_MODE = "cache"
LIVE_MODE = "live"


def _resolve_mode(use_cache: bool | None) -> str:
    if use_cache is not None:
        return CACHE_MODE if use_cache else LIVE_MODE
    return os.environ.get("PIPELINE_MODE", LIVE_MODE).lower()


def _has_cached_postings() -> bool:
    return _get_collection(POSTINGS_COLLECTION).count() > 0


async def run_pipeline(
    profile: dict | None = None,
    use_cache: bool | None = None,
    cosine_top_k: int = 30,
    rerank_top_k: int = 10,
    **search_params,
) -> list[dict]:
    """Full cascade: scrape/index (skipped in cache mode) -> cosine rank ->
    cross-encoder rerank.

    Args:
        use_cache: True = cache mode, False = live mode, None (default) =
            fall back to the PIPELINE_MODE env var.
        **search_params: forwarded to search_jobs when running live
            (searchTerm, location, resultsWanted, ...) - ignored in cache
            mode since no scrape happens.

    Returns:
        list[dict]: top rerank_top_k postings, cross-encoder scored.
    """
    profile = profile or load_profile()
    mode = _resolve_mode(use_cache)

    if mode == CACHE_MODE:
        if not _has_cached_postings():
            raise RuntimeError(
                "PIPELINE_MODE=cache but the 'postings' collection is empty - "
                "run in live mode at least once first so there's something to cache-rank."
            )
        logger.info("Cache mode: skipping JobSpy scrape + embedding, ranking persisted postings")
    else:
        logger.info("Live mode: scraping JobSpy and embedding live")
        postings = await search_jobs(**search_params)
        index_postings(postings)

    cosine_ranked = rank_postings_by_profile(profile, top_k=cosine_top_k)
    reranked = rerank_postings(profile, cosine_ranked, top_k=rerank_top_k)
    return reranked


async def _smoke_test():
    reranked = await run_pipeline(
        searchTerm="Full Stack Developer",
        location="Tel Aviv",
        countryIndeed="Israel",
        resultsWanted=50,
        siteNames="indeed,linkedin",
        hoursOld=168,
        format="json",
    )
    mode = _resolve_mode(None)
    table = Table(title=f"Pipeline output ({mode} mode)")
    table.add_column("Rerank score")
    table.add_column("Cosine similarity")
    table.add_column("Title")
    table.add_column("Company")
    for r in reranked:
        table.add_row(
            f"{r['rerank_score']:.3f}",
            f"{r['similarity']:.3f}",
            str(r["metadata"].get("title")),
            str(r["metadata"].get("company")),
        )
    console.print(table)


if __name__ == "__main__":
    asyncio.run(_smoke_test())
