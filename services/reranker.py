"""
reranker.py

description: Cross-encoder reranker for the retrieval cascade (Phase 2.6).
Phase 2.5's cosine ranking is a bi-encoder - profile and posting are embedded
separately, so the model never sees them together and can miss constraint
mismatches (e.g. "8+ years required" against a 4-year profile). A cross-
encoder feeds [profile] [SEP] [posting] through one transformer with full
cross-attention between them, trading precomputability for precision - so it
only runs on the cosine-narrowed shortlist, never the full scrape.

Cascade: 50 scraped -> cosine top 30 (services/vector_store) -> cross-encoder
top 10 (this module) -> Gemini reasons over the 10. Filtering only - no
score here ever writes to the sheet or decides a match.

usage: python -m services.reranker   # smoke test: run the full cascade
                                      # end to end against whatever is
                                      # already indexed in the vector store,
                                      # print cosine vs. rerank scores side
                                      # by side with per-stage timing
"""
import time

from rich.console import Console
from rich.table import Table
from sentence_transformers import CrossEncoder

from core.logger import get_logger
from services.vector_store import load_profile, rank_postings_by_profile

logger = get_logger(__name__)
console = Console()

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = CrossEncoder(CROSS_ENCODER_MODEL)


def _join(values) -> str | None:
    if not values:
        return None
    return "; ".join(str(v) for v in values)


def _reranker_query_text(profile: dict) -> str:
    """Target roles, must-haves, and key skills - not the raw CV dump."""
    prefs = profile.get("preferences", {})
    key_skills = [s for group in profile.get("skills", {}).values() for s in group]
    parts = [
        profile.get("title"),
        _join(prefs.get("target_roles")),
        _join(prefs.get("must_haves")),
        _join(key_skills),
    ]
    return "\n".join(p for p in parts if p)


def rerank_postings(profile: dict, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """Cross-encoder rerank a cosine-narrowed shortlist.

    Args:
        candidates: output of vector_store.rank_postings_by_profile - each a
            dict with at least {id, document, metadata, similarity}.

    Returns:
        list[dict]: top_k candidates, each with an added "rerank_score",
        sorted highest first.
    """
    if not candidates:
        logger.info("No candidates to rerank")
        return []
    query = _reranker_query_text(profile)
    pairs = [(query, c["document"]) for c in candidates]
    logger.info("Scoring %d pair(s) with cross-encoder", len(pairs))
    scores = _model.predict(pairs)
    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)
    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
    logger.info("Reranked to top %d", len(reranked))
    return reranked


def run_cascade(profile: dict, cosine_top_k: int = 30, rerank_top_k: int = 10) -> list[dict]:
    """Full retrieval cascade: cosine top_k -> cross-encoder top_k, with
    per-stage timing logged.
    """
    start = time.perf_counter()
    cosine_ranked = rank_postings_by_profile(profile, top_k=cosine_top_k)
    cosine_elapsed = time.perf_counter() - start
    logger.info("Cosine stage: %d candidate(s) in %.3fs", len(cosine_ranked), cosine_elapsed)

    start = time.perf_counter()
    reranked = rerank_postings(profile, cosine_ranked, top_k=rerank_top_k)
    rerank_elapsed = time.perf_counter() - start
    logger.info("Cross-encoder stage: %d candidate(s) in %.3fs", len(reranked), rerank_elapsed)

    return reranked


def _smoke_test():
    profile = load_profile()
    reranked = run_cascade(profile)
    if not reranked:
        console.print("[yellow]No postings in the vector store - run services.vector_store first.[/yellow]")
        return

    table = Table(title="Cosine vs. cross-encoder rankings")
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
    _smoke_test()
