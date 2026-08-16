"""
build_golden_set.py

description: Phase 8.1 - build the golden set that gates every other Phase 8
metric (8.2, 8.3), and Phase 9's classifier if that's ever picked up. Pulls a
varied batch of real postings across a few search queries, then walks
through each asking for a manual label:

  0 = bad match, 1 = plausible, 2 = strong match, s = skip, q = quit and save

This is real subjective judgment - there's no way to automate it - so this
script's only job is to make the ~40-label session (roughly an hour, per the
build guide) fast and resumable: results save incrementally to
eval/golden_set.json, and re-running skips postings already labeled.

usage: python -m eval.build_golden_set
       python -m eval.build_golden_set --refresh   # force a fresh scrape first
"""
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from core.logger import get_logger
from scripts.dev_cache import cached_search_jobs
from services.vector_store import _posting_id

console = Console()
logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN_SET_PATH = _REPO_ROOT / "eval" / "golden_set.json"

# A few distinct queries, not one - variety in role/seniority matters more
# for a golden set than raw volume from a single search.
_SEARCH_QUERIES = [
    dict(cache_name="golden_search_1", searchTerm="Full Stack Developer", location="Tel Aviv",
         countryIndeed="Israel", resultsWanted=30, siteNames="indeed,linkedin",
         hoursOld=336, format="json"),
    dict(cache_name="golden_search_2", searchTerm="Backend Engineer", location="Tel Aviv",
         countryIndeed="Israel", resultsWanted=30, siteNames="indeed,linkedin",
         hoursOld=336, format="json"),
]

_LABELS = {"0": 0, "1": 1, "2": 2}


def _load_existing() -> dict[str, dict]:
    if _GOLDEN_SET_PATH.exists():
        entries = json.loads(_GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        return {e["id"]: e for e in entries}
    return {}


def _save(entries: dict[str, dict]) -> None:
    _GOLDEN_SET_PATH.parent.mkdir(exist_ok=True)
    _GOLDEN_SET_PATH.write_text(json.dumps(list(entries.values()), indent=2), encoding="utf-8")


async def _gather_candidates(force_refresh: bool) -> list[dict]:
    all_postings: dict[str, dict] = {}
    for query in _SEARCH_QUERIES:
        postings = await cached_search_jobs(force_refresh=force_refresh, **query)
        for p in postings:
            all_postings[_posting_id(p)] = p
    return list(all_postings.values())


def _prompt_label(posting: dict) -> str:
    """Returns '0'/'1'/'2', 's' (skip), or 'q' (quit)."""
    console.print(Panel(
        f"[bold]{posting.get('title')}[/bold] @ {posting.get('company')}\n"
        f"{posting.get('location', '')}\n\n"
        f"{posting.get('description') or '(no description)'}",
        title="label this posting",
    ))
    while True:
        answer = console.input(
            "[bold]0[/bold]=bad  [bold]1[/bold]=plausible  [bold]2[/bold]=strong  "
            "[bold]s[/bold]=skip  [bold]q[/bold]=quit+save > "
        ).strip().lower()
        if answer in _LABELS or answer in ("s", "q"):
            return answer
        console.print("[red]enter 0, 1, 2, s, or q[/red]")


async def main():
    force_refresh = "--refresh" in sys.argv
    existing = _load_existing()
    console.print(f"[cyan]{len(existing)} posting(s) already labeled in {_GOLDEN_SET_PATH}[/cyan]")

    candidates = await _gather_candidates(force_refresh)
    unlabeled = [p for p in candidates if _posting_id(p) not in existing]
    console.print(f"[cyan]{len(candidates)} candidate posting(s) fetched, "
                  f"{len(unlabeled)} need labeling.[/cyan]")

    for posting in unlabeled:
        answer = _prompt_label(posting)
        if answer == "q":
            break
        if answer == "s":
            continue
        entry = {
            "id": _posting_id(posting),
            "title": posting.get("title"),
            "company": posting.get("company"),
            "description": posting.get("description"),
            "location": posting.get("location"),
            "link": posting.get("jobUrl") or posting.get("jobUrlDirect"),
            "label": _LABELS[answer],
        }
        existing[entry["id"]] = entry
        _save(existing)  # incremental - safe to Ctrl-C or 'q' at any point

    counts = {0: 0, 1: 0, 2: 0}
    for e in existing.values():
        counts[e["label"]] += 1
    console.print(f"[bold green]{len(existing)} posting(s) labeled total, "
                  f"saved to {_GOLDEN_SET_PATH}[/bold green]")
    console.print(f"label distribution: {counts}")
    if len(existing) < 40:
        console.print(f"[yellow]Guide target is ~40 labels; {40 - len(existing)} to go. "
                       f"Re-run this script (or add more queries) to keep labeling.[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
