"""
groundedness_eval.py

description: Phase 8.3 - measures whether Phase 4.5's RAG cover-letter
drafting actually reduces hallucination relative to a no-RAG baseline (whole
CV dumped into the prompt). For up to 20 golden-set postings worth applying
to (label >= 1), drafts a letter both ways via services.cover_letter (which
already supports use_rag=True/False), then uses Gemini as a judge to
decompose each letter into atomic factual claims and mark each grounded in
the context that variant was actually given - RAG: the retrieved CV chunks;
no-RAG: the full CV text - or not.

Expected result per the build guide: RAG matches or beats no-RAG on
groundedness at a fraction of the prompt tokens. Either outcome is
reportable - the point is having the number, not winning.

usage: python -m eval.groundedness_eval
       python -m eval.groundedness_eval --golden path/to/other_golden_set.json -n 10
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.table import Table

from core.constants import GEMINI_MODEL
from core.logger import get_logger
from services.cover_letter import _flatten_cv, draft_cover_letter
from services.vector_store import load_profile

load_dotenv()
console = Console()
logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_GOLDEN_PATH = _REPO_ROOT / "eval" / "golden_set.json"
_RESULTS_PATH = _REPO_ROOT / "eval" / "results" / "groundedness_results.json"
_SAMPLE_SIZE = 20

_JUDGE_PROMPT = """You are checking a cover letter for hallucination.

Context the letter was allowed to draw from:
{context}

Cover letter:
{letter}

Decompose the letter into atomic factual claims about the candidate's experience, skills, or \
background (ignore generic filler like "I am excited to apply"). For each claim, decide whether \
it is directly supported by the context above ("grounded": true) or not ("grounded": false).

Return ONLY a JSON array, no other text:
[{{"claim": "<claim text>", "grounded": true|false}}, ...]
"""

_judge_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=os.environ["GEMINI_API_KEY"])


def _load_golden(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist - run `python -m eval.build_golden_set` first.")
    entries = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in entries if e.get("label", 0) >= 1]


def _as_text(content) -> str:
    """ChatGoogleGenerativeAI sometimes returns content as a list of blocks
    rather than a plain string."""
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content


def _count_tokens(llm, text: str) -> int:
    try:
        return llm.get_num_tokens(text)
    except Exception:
        return len(text) // 4


def _judge_letter(letter: str, context: str) -> list[dict]:
    prompt = _JUDGE_PROMPT.format(context=context, letter=letter)
    response = _judge_llm.invoke(prompt)
    text = _as_text(response.content)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        logger.warning("Judge returned no JSON array; treating as zero claims. Response: %r", text[:200])
        return []
    return json.loads(text[start:end + 1])


def _evaluate_letter(posting: dict, profile: dict, use_rag: bool) -> dict:
    result = draft_cover_letter(posting, profile, use_rag=use_rag)
    letter = result["letter"]
    context = "\n".join(f"- {c['text']}" for c in result["chunks"]) if use_rag else _flatten_cv(profile)

    claims = _judge_letter(letter, context)
    grounded = sum(1 for c in claims if c.get("grounded"))
    total = len(claims)

    return {
        "posting_id": posting["id"],
        "grounded": grounded,
        "total_claims": total,
        "pct_grounded": (grounded / total) if total else None,
        "prompt_tokens": _count_tokens(_judge_llm, result["prompt"]),
    }


def _print_table(variant_results: dict[str, list[dict]]) -> None:
    table = Table(title="RAG vs. no-RAG cover-letter groundedness")
    for col in ["Variant", "Letters", "Avg % grounded", "Avg prompt tokens"]:
        table.add_column(col)
    for name, results in variant_results.items():
        scored = [r for r in results if r["total_claims"] > 0]
        avg_pct = sum(r["pct_grounded"] for r in scored) / len(scored) if scored else 0.0
        avg_tokens = sum(r["prompt_tokens"] for r in results) / len(results) if results else 0.0
        table.add_row(name, str(len(results)), f"{avg_pct:.1%}", f"{avg_tokens:.0f}")
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Phase 8.3 groundedness evaluation")
    parser.add_argument("--golden", type=Path, default=_DEFAULT_GOLDEN_PATH)
    parser.add_argument("-n", "--sample-size", type=int, default=_SAMPLE_SIZE)
    args = parser.parse_args()

    golden = _load_golden(args.golden)[:args.sample_size]
    if not golden:
        raise ValueError(
            "No label>=1 postings in the golden set - label some worth applying to "
            "with eval.build_golden_set first."
        )
    profile = load_profile()
    console.print(f"[cyan]Evaluating groundedness over {len(golden)} posting(s), RAG vs. no-RAG[/cyan]")

    variant_results: dict[str, list[dict]] = {"RAG": [], "no-RAG": []}
    for posting in golden:
        logger.info("Drafting letters for %r", posting.get("title"))
        variant_results["RAG"].append(_evaluate_letter(posting, profile, use_rag=True))
        variant_results["no-RAG"].append(_evaluate_letter(posting, profile, use_rag=False))

    _print_table(variant_results)

    _RESULTS_PATH.parent.mkdir(exist_ok=True, parents=True)
    _RESULTS_PATH.write_text(json.dumps(variant_results, indent=2), encoding="utf-8")
    console.print(f"[green]Saved results to {_RESULTS_PATH}[/green]")


if __name__ == "__main__":
    main()
