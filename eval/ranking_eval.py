"""
ranking_eval.py

description: Phase 8.2 - scores four retrieval variants against the golden
set (eval/golden_set.json) with classic IR metrics (precision@10, nDCG@10,
MRR), plus latency and token cost, so "I added a reranker" becomes a real
before/after number.

  V0 - BM25 keyword baseline (rank_bm25)
  V1 - bi-encoder cosine, computed directly from services.vector_store's
       embedding model (not the persistent Chroma collection - keeps this
       eval isolated from whatever happens to be indexed there at runtime)
  V2 - V1 -> cross-encoder rerank (services.reranker.rerank_postings, reused)
  V3 - LLM-only: Gemini scores every golden posting directly in one call

V3 may well win on raw quality - the interesting finding is what it costs
relative to V2 (see the "tokens" column), and whether the gap justifies it.

usage: python -m eval.ranking_eval
       python -m eval.ranking_eval --golden path/to/other_golden_set.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table

from core.constants import GEMINI_MODEL
from core.logger import get_logger
from eval.metrics import mrr, ndcg_at_k, precision_at_k
from services.reranker import rerank_postings
from services.vector_store import _posting_embed_text, _profile_query_text, load_profile

load_dotenv()
console = Console()
logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_GOLDEN_PATH = _REPO_ROOT / "eval" / "golden_set.json"
_RESULTS_PATH = _REPO_ROOT / "eval" / "results" / "ranking_results.json"
_TOP_K = 10

_LLM_ONLY_PROMPT = """You are scoring job postings for fit against a candidate profile.

Candidate profile (target roles, must-haves, key skills):
{profile_text}

For each posting below, output an integer relevance score from 0 to 100 (100 = perfect fit, \
0 = no fit at all). Return ONLY a JSON array, one object per posting, no other text:
[{{"id": "<id>", "score": <int>}}, ...]

Postings:
{postings_block}
"""


def _load_golden(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet - run `python -m eval.build_golden_set` first."
        )
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not entries:
        raise ValueError(f"{path} is empty - label some postings with eval.build_golden_set first.")
    return entries[:limit] if limit else entries


def _labels_in_rank_order(ranked_ids: list[str], golden_by_id: dict[str, dict]) -> list[int]:
    return [golden_by_id[pid]["label"] if pid in golden_by_id else 0 for pid in ranked_ids]


# ---------------------------------------------------------------------------
# V0 - BM25
# ---------------------------------------------------------------------------

def score_bm25(golden: list[dict], profile: dict) -> tuple[list[str], float, dict]:
    start = time.perf_counter()
    tokenized_docs = [_posting_embed_text(g).lower().split() for g in golden]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = _profile_query_text(profile).lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked_ids = [g["id"] for g, _ in sorted(zip(golden, scores), key=lambda pair: pair[1], reverse=True)]
    elapsed = time.perf_counter() - start
    return ranked_ids, elapsed, {}


# ---------------------------------------------------------------------------
# V1 - cosine (bi-encoder)
# ---------------------------------------------------------------------------

def _cosine_scores(golden: list[dict], profile: dict) -> list[float]:
    # Cached by id (scripts.dev_cache) - the golden set is 40-100+ postings,
    # comfortably enough to exhaust a free-tier embedding quota mid-run, and
    # a rerun should only pay for postings it hasn't embedded yet.
    from scripts.dev_cache import cached_embed_documents, cached_profile_query_embedding

    embeddings_by_id = cached_embed_documents(
        "golden_set_postings", [(g["id"], _posting_embed_text(g)) for g in golden]
    )
    doc_vectors = np.array([embeddings_by_id[g["id"]] for g in golden])
    query_vector = np.array(cached_profile_query_embedding(profile))
    doc_norms = np.linalg.norm(doc_vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)
    return list((doc_vectors @ query_vector) / (doc_norms * query_norm + 1e-12))


def score_cosine(golden: list[dict], profile: dict) -> tuple[list[str], float, dict]:
    start = time.perf_counter()
    scores = _cosine_scores(golden, profile)
    ranked_ids = [g["id"] for g, _ in sorted(zip(golden, scores), key=lambda pair: pair[1], reverse=True)]
    elapsed = time.perf_counter() - start
    return ranked_ids, elapsed, {}


# ---------------------------------------------------------------------------
# V2 - cosine -> cross-encoder rerank
# ---------------------------------------------------------------------------

def score_cosine_cross_encoder(golden: list[dict], profile: dict) -> tuple[list[str], float, dict]:
    start = time.perf_counter()
    cosine_scores = _cosine_scores(golden, profile)
    candidates = [
        {"id": g["id"], "document": _posting_embed_text(g), "metadata": g, "similarity": s}
        for g, s in zip(golden, cosine_scores)
    ]
    reranked = rerank_postings(profile, candidates, top_k=len(candidates))
    ranked_ids = [c["id"] for c in reranked]
    elapsed = time.perf_counter() - start
    return ranked_ids, elapsed, {}


# ---------------------------------------------------------------------------
# V3 - LLM-only
# ---------------------------------------------------------------------------

def _as_text(content) -> str:
    """ChatGoogleGenerativeAI sometimes returns content as a list of blocks
    (e.g. [{"type": "text", "text": "..."}]) rather than a plain string."""
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content


def _parse_llm_scores(content) -> dict[str, float]:
    text = _as_text(content)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"Could not find a JSON array in LLM response: {text[:200]!r}")
    parsed = json.loads(text[start:end + 1])
    return {item["id"]: item["score"] for item in parsed}


def score_llm_only(golden: list[dict], profile: dict) -> tuple[list[str], float, dict]:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=os.environ["GEMINI_API_KEY"])
    postings_block = "\n\n".join(
        f"id: {g['id']}\ntitle: {g.get('title')}\ncompany: {g.get('company')}\n"
        f"description: {(g.get('description') or '')[:500]}"
        for g in golden
    )
    prompt = _LLM_ONLY_PROMPT.format(profile_text=_profile_query_text(profile), postings_block=postings_block)

    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    scores = _parse_llm_scores(response.content)
    ranked_ids = [g["id"] for g in sorted(golden, key=lambda g: scores.get(g["id"], 0), reverse=True)]
    usage = dict(getattr(response, "usage_metadata", None) or {})
    return ranked_ids, elapsed, usage


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_VARIANTS = {
    "V0 BM25": score_bm25,
    "V1 cosine": score_cosine,
    "V2 cosine+cross-encoder": score_cosine_cross_encoder,
    "V3 LLM-only": score_llm_only,
}


def evaluate_variant(name: str, scorer, golden: list[dict], profile: dict, golden_by_id: dict) -> dict:
    logger.info("Scoring variant: %s", name)
    ranked_ids, elapsed, usage = scorer(golden, profile)
    labels = _labels_in_rank_order(ranked_ids, golden_by_id)
    return {
        "variant": name,
        "precision@10": precision_at_k(labels, _TOP_K),
        "ndcg@10": ndcg_at_k(labels, _TOP_K),
        "mrr": mrr(labels),
        "latency_s": elapsed,
        "tokens": usage.get("total_tokens", 0),
        "top_10_ids": ranked_ids[:_TOP_K],
    }


def _print_table(results: list[dict]) -> None:
    table = Table(title="Retrieval cascade: quality vs. cost")
    for col in ["Variant", "Precision@10", "nDCG@10", "MRR", "Latency (s)", "Tokens"]:
        table.add_column(col)
    for r in results:
        table.add_row(
            r["variant"], f"{r['precision@10']:.3f}", f"{r['ndcg@10']:.3f}", f"{r['mrr']:.3f}",
            f"{r['latency_s']:.3f}", str(r["tokens"] or "-"),
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Phase 8.2 ranking evaluation")
    parser.add_argument("--golden", type=Path, default=_DEFAULT_GOLDEN_PATH)
    parser.add_argument("-n", "--limit", type=int, default=None,
                         help="only evaluate the first N golden-set postings (useful to stay under quota)")
    args = parser.parse_args()

    golden = _load_golden(args.golden, limit=args.limit)
    golden_by_id = {g["id"]: g for g in golden}
    profile = load_profile()
    console.print(f"[cyan]Evaluating {len(golden)} golden-set posting(s) "
                  f"across {len(_VARIANTS)} variant(s)[/cyan]")

    results = []
    for name, scorer in _VARIANTS.items():
        try:
            results.append(evaluate_variant(name, scorer, golden, profile, golden_by_id))
        except Exception as exc:
            if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                console.print(
                    f"[red]{name}: Gemini quota exhausted (429) - skipping this variant.[/red]\n"
                    f"[yellow]Embeddings/scores computed before the failure are cached under "
                    f"dev_cache/ - re-run this command later (once quota resets) to pick up "
                    f"where it left off.[/yellow]"
                )
            else:
                console.print(f"[red]{name}: failed - {exc}[/red]")
            logger.exception("%s failed", name)

    if not results:
        console.print("[red]No variant completed - nothing to report.[/red]")
        return

    _print_table(results)

    _RESULTS_PATH.parent.mkdir(exist_ok=True, parents=True)
    _RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"[green]Saved results to {_RESULTS_PATH}[/green]")


if __name__ == "__main__":
    main()
