"""
dev_cache.py

description: Dev-only cache for JobSpy scrape results. Iterating on Phase
4/4.5/5 means re-running search_jobs repeatedly, and JobSpy is scraping under
the hood (Cloudflare/CAPTCHA-prone, no official rate-limit budget) - so
during development, scrape once and replay from disk instead of re-hitting
the boards on every run.

Not used by any agent tool or production path (agent/tools.py always calls
the live integrations.jobs_searching.search_jobs) - only by smoke tests and
manual dev runs that opt in explicitly. Cached files live in dev_cache/
(gitignored) and are plain JSON, gitignored.

Posting embeddings don't need a separate cache here: services/vector_store.py
already persists vectors to ./vector_store/ and (as of index_postings'
skip_existing option) won't re-embed postings whose id is already indexed.
The one embedding that collection-based caching doesn't cover is the
profile-query vector - a single vector, not a batch of documents keyed by
id - so it gets its own flat-file cache below (dev_cache/profile_query.json),
keyed by an equality check against the profile text itself rather than a
Chroma collection: no ANN index is needed for one vector.

usage: python -m scripts.dev_cache                    # cache a default search
       python -m scripts.dev_cache --refresh           # force a fresh scrape
"""
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console

from core.logger import get_logger

console = Console()
logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _REPO_ROOT / "dev_cache"

_DEFAULT_SEARCH = dict(
    searchTerm="Full Stack Developer",
    location="Tel Aviv",
    countryIndeed="Israel",
    resultsWanted=50,
    siteNames="indeed,linkedin",
    hoursOld=168,
    format="json",
)


def _cache_path(name: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"{name}.json"


async def cached_search_jobs(cache_name: str = "postings", force_refresh: bool = False,
                              **params) -> list[dict]:
    """Return postings from dev_cache/<cache_name>.json if present, else scrape
    once via the live search_jobs and save the result.

    Args:
        cache_name: cache file stem under dev_cache/.
        force_refresh: ignore any existing cache file and re-scrape.
        **params: forwarded to integrations.jobs_searching.search_jobs on a
            cache miss (ignored on a cache hit).

    Returns:
        list[dict]: postings, same shape as search_jobs.
    """
    path = _cache_path(cache_name)
    if path.exists() and not force_refresh:
        postings = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Loaded %d cached posting(s) from %s", len(postings), path)
        return postings

    from integrations.jobs_searching import search_jobs
    postings = await search_jobs(**params)
    path.write_text(json.dumps(postings, indent=2), encoding="utf-8")
    logger.info("Scraped %d posting(s), cached to %s", len(postings), path)
    return postings


def cached_profile_query_embedding(profile: dict, force_refresh: bool = False) -> list[float]:
    """Return the embedding of the profile's query text from
    dev_cache/profile_query.json if the cached text still matches the
    current profile, else embed it live and refresh the cache.

    Args:
        force_refresh: ignore any existing cache entry and re-embed.

    Returns:
        list[float]: embedding of vector_store._profile_query_text(profile).
    """
    from services.vector_store import _embeddings, _profile_query_text

    text = _profile_query_text(profile)
    path = _cache_path("profile_query")
    if path.exists() and not force_refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("text") == text:
            logger.info("Loaded cached profile-query embedding from %s", path)
            return cached["embedding"]
        logger.info("Profile query text changed; cache stale, re-embedding")

    vector = _embeddings.embed_query(text)
    path.write_text(json.dumps({"text": text, "embedding": vector}), encoding="utf-8")
    logger.info("Embedded profile-query text, cached to %s", path)
    return vector


async def _main():
    force_refresh = "--refresh" in sys.argv
    postings = await cached_search_jobs(force_refresh=force_refresh, **_DEFAULT_SEARCH)
    console.print(f"[green]{len(postings)} posting(s) available in dev_cache/postings.json[/green]")


if __name__ == "__main__":
    asyncio.run(_main())
