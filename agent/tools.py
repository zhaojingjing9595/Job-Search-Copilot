"""
tools.py

description: Phase 4 - wraps the standalone functions built in earlier phases
as LangChain `@tool`s so an LLM can call them. This is glue, not new logic:
docstrings here are what the model reads to decide when/how to call each
tool, so they describe the *agent-facing* contract, not the implementation.

`search_jobs` returns the retrieval-cascade output (cosine top 30 -> cross-
encoder top 10, services/vector_store.py + services/reranker.py) rather than
the raw JobSpy scrape - the agent only ever sees the narrowed shortlist, and
has no way to ask for more without a new search.

usage: python -m agent.tools   # standalone smoke test: list the bound tools
"""
import asyncio
import json
from datetime import date

from langchain_core.tools import tool
from rich.console import Console

from core.logger import get_logger
from integrations.jobs_searching import search_jobs as _jobspy_search
from integrations.sheets_logging import log_job_match
from services.cover_letter import draft_cover_letter as _draft_cover_letter
from services.vector_store import index_postings, load_profile
from services.reranker import run_cascade

console = Console()
logger = get_logger(__name__)


def _format_posting(candidate: dict) -> dict:
    """Trim a cascade candidate (id/document/metadata/similarity/rerank_score)
    down to the fields the LLM actually needs to reason about a match."""
    meta = candidate["metadata"]
    return {
        "id": candidate["id"],
        "title": meta.get("title"),
        "company": meta.get("company"),
        "location": meta.get("location"),
        "description": meta.get("description"),
        "job_type": meta.get("jobType"),
        "job_level": meta.get("jobLevel"),
        "link": meta.get("jobUrl") or meta.get("jobUrlDirect"),
        "cosine_similarity": round(candidate.get("similarity", 0.0), 4),
        "rerank_score": round(candidate.get("rerank_score", 0.0), 4),
    }


@tool
async def search_jobs(search_term: str, location: str = "Tel Aviv",
                       country_indeed: str = "Israel", results_wanted: int = 50,
                       hours_old: int = 168) -> str:
    """Search live job postings and return the top matches for the candidate's profile.

    Runs the full retrieval cascade: scrapes up to `results_wanted` postings,
    cosine-ranks them against the candidate's profile (top 30), then cross-
    encoder-reranks that shortlist (top 10). Only the final top 10 are
    returned - this is a shortlist to reason over, not the full scrape.

    Args:
        search_term: role/keywords to search for, e.g. "Full Stack Engineer".
        location: city/region to search in.
        country_indeed: country code Indeed should search within.
        results_wanted: max postings to scrape before ranking (20-50 typical).
        hours_old: only include postings newer than this many hours.

    Returns:
        JSON array of up to 10 postings, each with title, company, location,
        description, link, job_type, job_level, cosine_similarity, and
        rerank_score. Higher rerank_score means a better fit.
    """
    postings = await _jobspy_search(
        searchTerm=search_term,
        location=location,
        countryIndeed=country_indeed,
        resultsWanted=results_wanted,
        hoursOld=hours_old,
        format="json",
    )
    if not postings:
        return json.dumps([])

    index_postings(postings)
    profile = load_profile()
    shortlist = run_cascade(profile, cosine_top_k=30, rerank_top_k=10)
    return json.dumps([_format_posting(c) for c in shortlist])


@tool
def get_profile() -> str:
    """Return the candidate's profile: title, summary, experience, skills, and preferences.

    Use this to ground match reasoning and cover-letter drafting in the
    candidate's actual background rather than assuming details.

    Returns:
        JSON object matching profile.json.
    """
    return json.dumps(load_profile())


@tool
def draft_cover_letter(title: str, company: str, description: str) -> str:
    """Draft a cover letter for a specific posting, grounded in the candidate's CV.

    Retrieves the CV bullets most relevant to this posting and writes the
    letter using only those bullets - it will not invent experience the
    candidate doesn't have. Call this only for postings worth applying to.

    Args:
        title: the posting's job title.
        company: the posting's company name.
        description: the posting's job description text.

    Returns:
        JSON object {"letter": str, "evidence": list[str]} - evidence is the
        exact CV bullets the letter is grounded in.
    """
    result = _draft_cover_letter({"title": title, "company": company, "description": description})
    return json.dumps({
        "letter": result["letter"],
        "evidence": [c["text"] for c in result["chunks"]],
    })


@tool
def log_posting(company: str, title: str, match_reasoning: str, status: str,
                 link: str, log_date: str = "") -> str:
    """Append one evaluated posting to the tracking sheet.

    Only call this for postings you have actually evaluated and reasoned
    about - match_reasoning must explain *why*, not restate a score. Never
    call this for postings you haven't looked at.

    Guardrails applied automatically: a posting whose link is already in the
    sheet is skipped rather than logged twice (report this, don't retry).

    Args:
        company: company name.
        title: job title.
        match_reasoning: your explanation of why this is (or isn't) a good match.
        status: e.g. "to apply", "applied", "not a fit".
        link: posting URL.
        log_date: ISO date string; defaults to today if omitted.

    Returns:
        "appended", "skipped_duplicate" (link already logged), or "dry_run"
        (DRY_RUN is set - nothing was written).
    """
    row = {
        "company": company,
        "title": title,
        "match_reasoning": match_reasoning,
        "status": status,
        "link": link,
        "date": log_date or date.today().isoformat(),
    }
    return log_job_match(row)


AGENT_TOOLS = [search_jobs, get_profile, draft_cover_letter, log_posting]


def _smoke_test():
    for t in AGENT_TOOLS:
        console.print(f"[bold cyan]{t.name}[/bold cyan]: {t.description.splitlines()[0]}")
        console.print(f"  args: {t.args}")


if __name__ == "__main__":
    _smoke_test()
