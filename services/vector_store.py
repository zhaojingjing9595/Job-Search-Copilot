"""
vector_store.py

description: Chroma wrapper for the semantic layer that sits in front of the
agent (Phase 2.5). Two persistent collections, both cosine-indexed:

  cv_chunks  - CV highlight bullets, for RAG cover-letter drafting later
  postings   - scraped job postings, for cosine pre-ranking + near-dup detection

Embeddings come from GoogleGenerativeAIEmbeddings (gemini-embedding-001),
reusing the existing Gemini key - no new credential needed.

Posting documents keep only the fields that carry signal about role fit
(title, company, summary/description, skills, level, type, experience,
location) so the embedding isn't diluted by noise. Every other field
(salary, urls, isRemote, company stats, etc.) is kept as Chroma metadata -
retrievable for filtering/dedup/sheet-logging, but not embedded.

usage: python -m services.vector_store   # standalone smoke test: index the CV,
                                       # pull a live JobSpy batch, rank + dedup it
"""
import asyncio
import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rich.console import Console
from rich.table import Table

from core.constants import GEMINI_EMBEDDING_MODEL
from core.logger import get_logger

load_dotenv()
console = Console()
logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VECTOR_STORE_DIR = _REPO_ROOT / "vector_store"
_PROFILE_PATH = _REPO_ROOT / "profile.json"

CV_CHUNKS_COLLECTION = "cv_chunks"
POSTINGS_COLLECTION = "postings"
DEFAULT_DUPLICATE_THRESHOLD = 0.92

_embeddings = GoogleGenerativeAIEmbeddings(
    model=GEMINI_EMBEDDING_MODEL, google_api_key=os.environ["GEMINI_API_KEY"]
)
_client = chromadb.PersistentClient(path=str(_VECTOR_STORE_DIR))


def _get_collection(name: str):
    # cosine space, since ranking and dedup below both reason in cosine similarity
    return _client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def load_profile(path: Path = _PROFILE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _join(values) -> str | None:
    if not values:
        return None
    return "; ".join(str(v) for v in values)


# ---------------------------------------------------------------------------
# CV chunks
# ---------------------------------------------------------------------------

def chunk_cv(profile: dict) -> list[dict]:
    """Split profile['professional_experiences'] into bullet-level chunks.

    Returns:
        list[dict]: each with id, text, and metadata (role, company, date_range).
    """
    chunks = []
    for exp in profile.get("professional_experiences", []):
        company = exp.get("company", "")
        role = exp.get("role", "")
        date_range = exp.get("years", "")
        for i, highlight in enumerate(exp.get("highlights", [])):
            chunk_id = f"{company}-{role}-{i}".replace(" ", "_")
            chunks.append({
                "id": chunk_id,
                "text": highlight,
                "metadata": {"role": role, "company": company, "date_range": date_range},
            })
    return chunks


def index_cv_chunks(profile: dict | None = None, skip_existing: bool = True) -> int:
    """Embed and upsert CV highlight chunks into cv_chunks.

    Args:
        skip_existing: don't re-embed chunks whose id is already in the
            collection - the CV doesn't change mid dev-session, so reruns
            can skip the embedding call entirely, same as index_postings.

    Returns:
        int: number of chunks newly embedded (0 if all were already indexed).
    """
    profile = profile or load_profile()
    chunks = chunk_cv(profile)
    if not chunks:
        logger.info("No CV chunks to index")
        return 0

    collection = _get_collection(CV_CHUNKS_COLLECTION)
    ids = [c["id"] for c in chunks]
    existing_ids = set(collection.get(ids=ids)["ids"]) if skip_existing else set()
    new = [c for c in chunks if c["id"] not in existing_ids]
    if not new:
        logger.info("All %d CV chunk(s) already indexed; skipped re-embedding", len(chunks))
        return 0

    logger.info("Embedding and indexing %d new CV chunk(s) (%d already indexed)",
                len(new), len(existing_ids))
    try:
        vectors = _embeddings.embed_documents([c["text"] for c in new])
        collection.upsert(
            ids=[c["id"] for c in new],
            embeddings=vectors,
            documents=[c["text"] for c in new],
            metadatas=[c["metadata"] for c in new],
        )
    except Exception:
        logger.exception("Failed to index CV chunks")
        raise
    logger.info("Indexed %d CV chunk(s)", len(new))
    return len(new)


# ---------------------------------------------------------------------------
# Postings
# ---------------------------------------------------------------------------

def _posting_id(posting: dict) -> str:
    return (
        posting.get("id")
        or posting.get("jobUrl")
        or posting.get("jobUrlDirect")
        or f"{posting.get('company', '')}-{posting.get('title', '')}"
    )


def _posting_embed_text(posting: dict) -> str:
    # Field names match JobSpy's own output columns (title, company, skills,
    # experienceRange, ...) - not the aspirational jobSpySchema.js in the
    # jobspy-mcp-server repo, which the server does not actually apply.
    parts = [
        posting.get("title"),
        posting.get("company"),
        posting.get("description"),
        _join(posting.get("skills")),
        posting.get("jobLevel"),
        posting.get("jobType"),
        posting.get("experienceRange"),
        posting.get("location"),
    ]
    return "\n".join(p for p in parts if p)


def _posting_metadata(posting: dict) -> dict:
    """Every field, flattened to Chroma-safe scalars (no None, no lists)."""
    metadata = {}
    for key, value in posting.items():
        if value is None:
            continue
        if isinstance(value, list):
            if not value:
                continue
            metadata[key] = "; ".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = str(value)
    return metadata


def index_postings(postings: list[dict], skip_existing: bool = True) -> int:
    """Embed and upsert scraped postings into the postings collection.

    Args:
        skip_existing: don't re-embed postings whose id is already in the
            collection. Chroma persists vectors across runs, but without this
            every call still burns embedding quota re-embedding the same
            postings - this makes reruns during development free.

    Returns:
        int: number of postings newly embedded (0 if all were already indexed).
    """
    if not postings:
        logger.info("No postings to index")
        return 0
    ids = [_posting_id(p) for p in postings]
    collection = _get_collection(POSTINGS_COLLECTION)

    existing_ids = set(collection.get(ids=ids)["ids"]) if skip_existing else set()
    new = [(i, p) for i, p in zip(ids, postings) if i not in existing_ids]
    if not new:
        logger.info("All %d posting(s) already indexed; skipped re-embedding", len(postings))
        return 0

    logger.info("Embedding and indexing %d new posting(s) (%d already indexed)",
                len(new), len(existing_ids))
    try:
        new_ids = [i for i, _ in new]
        texts = [_posting_embed_text(p) for _, p in new]
        vectors = _embeddings.embed_documents(texts)
        collection.upsert(
            ids=new_ids,
            embeddings=vectors,
            documents=texts,
            metadatas=[_posting_metadata(p) for _, p in new],
        )
    except Exception:
        logger.exception("Failed to index postings")
        raise
    logger.info("Indexed %d new posting(s)", len(new))
    return len(new)


# ---------------------------------------------------------------------------
# Ranking + near-duplicate detection
# ---------------------------------------------------------------------------

def _profile_query_text(profile: dict) -> str:
    prefs = profile.get("preferences", {})
    parts = [
        _join(prefs.get("target_roles")),
        _join(prefs.get("must_haves")),
        profile.get("title"),
        profile.get("summary"),
    ]
    return "\n".join(p for p in parts if p)


def rank_postings_by_profile(
    profile: dict, top_k: int = 30, query_vector: list[float] | None = None
) -> list[dict]:
    """Cosine-rank every posting in the postings collection against the
    profile's target-role text.

    Args:
        query_vector: precomputed embedding of _profile_query_text(profile) -
            pass this to skip the live embed call (e.g. from a dev-time
            cache in scripts/dev_cache.py). Defaults to None, which embeds
            live every call - the correct default for production, where the
            profile embedding should always reflect the current profile.

    Returns:
        list[dict]: top_k postings as {id, similarity, metadata, document},
        highest similarity first.
    """
    collection = _get_collection(POSTINGS_COLLECTION)
    count = collection.count()
    if count == 0:
        logger.info("No postings in the collection to rank")
        return []
    query_vector = query_vector or _embeddings.embed_query(_profile_query_text(profile))
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
    )
    ranked = []
    for i, posting_id in enumerate(result["ids"][0]):
        ranked.append({
            "id": posting_id,
            "similarity": 1 - result["distances"][0][i],
            "metadata": result["metadatas"][0][i],
            "document": result["documents"][0][i],
        })
    logger.info("Ranked %d posting(s) against profile", len(ranked))
    return ranked


def find_near_duplicates(threshold: float = DEFAULT_DUPLICATE_THRESHOLD) -> list[tuple[str, str, float]]:
    """Flag posting pairs above the cosine threshold as the same role
    cross-posted to different boards.

    Returns:
        list[tuple[str, str, float]]: (id_a, id_b, similarity), each pair once.
    """
    collection = _get_collection(POSTINGS_COLLECTION)
    total = collection.count()
    if total < 2:
        logger.info("Fewer than 2 postings in the collection; skipping dedup")
        return []
    all_postings = collection.get(include=["embeddings"])
    seen = set()
    pairs = []
    for i, posting_id in enumerate(all_postings["ids"]):
        neighbors = collection.query(
            query_embeddings=[all_postings["embeddings"][i]],
            n_results=min(5, total),
        )
        for j, neighbor_id in enumerate(neighbors["ids"][0]):
            if neighbor_id == posting_id:
                continue
            similarity = 1 - neighbors["distances"][0][j]
            if similarity < threshold:
                continue
            pair_key = tuple(sorted((posting_id, neighbor_id)))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            pairs.append((pair_key[0], pair_key[1], similarity))
    logger.info("Found %d near-duplicate pair(s) above threshold %.3f", len(pairs), threshold)
    return pairs


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

async def _smoke_test():
    from scripts.dev_cache import cached_search_jobs

    profile = load_profile()
    n_cv = index_cv_chunks(profile)
    console.print(f"[green]Indexed {n_cv} CV chunk(s).[/green]")

    # Cached to dev_cache/postings.json after the first run - avoids re-hitting
    # JobSpy's scrape (and its rate limits) on every dev iteration. Delete the
    # file, or pass force_refresh=True, to get a fresh scrape.
    postings = await cached_search_jobs(
        searchTerm="Full Stack engineer",
        location="Tel Aviv",
        countryIndeed="Israel",
        resultsWanted=50,
        format="json",
    )
    console.print(f"[cyan]Fetched {len(postings)} posting(s) (cache-or-scrape).[/cyan]")

    n_postings = index_postings(postings)
    console.print(f"[green]Indexed {n_postings} posting(s).[/green]")

    ranked = rank_postings_by_profile(profile, top_k=30)
    table = Table(title="Top postings by cosine similarity to profile")
    table.add_column("Similarity")
    table.add_column("Title")
    table.add_column("Company")
    for r in ranked[:10]:
        table.add_row(
            f"{r['similarity']:.3f}",
            str(r["metadata"].get("title")),
            str(r["metadata"].get("company")),
        )
    console.print(table)

    duplicates = find_near_duplicates()
    console.print(f"[yellow]Found {len(duplicates)} near-duplicate pair(s) above threshold.[/yellow]")
    for a, b, sim in duplicates:
        console.print(f"  {a}  <->  {b}   similarity={sim:.3f}")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
