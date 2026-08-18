"""
cv_profile.py

description: RAG-grounded customized CV profile drafting (Phase 4.5). Phases
2.5/2.6 are vector *search* - nothing retrieved conditions a generation. This
is what makes "RAG" accurate: the JD is embedded, the top-k `cv_chunks` are
retrieved from the vector store (services/vector_store.py), and only those
chunks are used to build a tailored profile. The profile is returned
alongside the chunks that produced it, so every draft ships with its
evidence and nothing in it should trace back to anything else.

A no-RAG variant (whole CV dumped into the prompt) is kept behind a flag
purely so Phase 8 groundedness evaluation has a baseline to compare against.

usage: python -m services.cv_profile   # smoke test against whatever
                                        # posting/profile is already indexed
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from rich.console import Console

from core.constants import GEMINI_MODEL
from core.logger import get_logger
from services.vector_store import CV_CHUNKS_COLLECTION, _embeddings, _get_collection, load_profile

load_dotenv()
console = Console()
logger = get_logger(__name__)

DEFAULT_TOP_K = 8
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SMOKE_TEST_OUTPUT_PATH = _REPO_ROOT / "dev_cache" / "cv_profile_smoke_test.json"
_APPLICATIONS_DIR = _REPO_ROOT / "applications"

_SCHEMA_EXAMPLE = """{
  "summary": "2-3 sentence professional summary rewritten to speak to this posting",
  "highlights": ["tailored, reworded experience bullet", "..."],
  "skills": ["skill relevant to this posting", "..."]
}"""

_PROMPT_TEMPLATE = """You are tailoring a CV profile for {name}, applying to the {title} role at {company}.

Job description:
{job_description}

Use ONLY the experience bullets below - do not invent experience, employers, or skills that aren't in them. \
If the posting asks for something none of these bullets cover, omit it rather than papering over the gap.

Experience bullets:
{bullets}

Return a customized CV profile as a JSON object grounded entirely in the bullets above: a rewritten \
professional summary tailored to this posting, a reordered/reworded set of highlight bullets (most relevant \
first), and a subset of skills relevant to this posting. Follow this schema: {schema}
Return only valid JSON, no extra commentary."""

_NO_RAG_PROMPT_TEMPLATE = """You are tailoring a CV profile for {name}, applying to the {title} role at {company}.

Job description:
{job_description}

Full candidate background:
{full_cv}

Return a customized CV profile as a JSON object: a rewritten professional summary tailored to this posting, \
a reordered/reworded set of highlight bullets (most relevant first), and a subset of skills relevant to this \
posting. Follow this schema: {schema}
Return only valid JSON, no extra commentary."""


def _posting_query_text(posting: dict) -> str:
    parts = [posting.get("title"), posting.get("company"), posting.get("description")]
    return "\n".join(p for p in parts if p)


def retrieve_cv_chunks(posting: dict, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Embed the JD and retrieve the top_k most relevant CV chunks.

    Returns:
        list[dict]: each {id, text, metadata, similarity}, highest similarity first.
    """
    collection = _get_collection(CV_CHUNKS_COLLECTION)
    count = collection.count()
    if count == 0:
        logger.info("No CV chunks indexed; run services.vector_store.index_cv_chunks first")
        return []
    query_vector = _embeddings.embed_query(_posting_query_text(posting))
    result = collection.query(query_embeddings=[query_vector], n_results=min(top_k, count))
    chunks = []
    for i, chunk_id in enumerate(result["ids"][0]):
        chunks.append({
            "id": chunk_id,
            "text": result["documents"][0][i],
            "metadata": result["metadatas"][0][i],
            "similarity": 1 - result["distances"][0][i],
        })
    logger.info("Retrieved %d CV chunk(s) for posting %r", len(chunks), posting.get("title"))
    return chunks


def _flatten_cv(profile: dict) -> str:
    lines = []
    for exp in profile.get("professional_experiences", []):
        lines.append(f"{exp.get('role')} at {exp.get('company')} ({exp.get('years')})")
        lines.extend(f"- {h}" for h in exp.get("highlights", []))
    return "\n".join(lines)


def _parse_profile_json(raw: str) -> dict:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.exception("LLM returned invalid JSON for CV profile")
        raise


def draft_cv_profile(posting: dict, profile: dict | None = None, top_k: int = DEFAULT_TOP_K,
                      use_rag: bool = True) -> dict:
    """Draft a customized CV profile for one posting.

    Args:
        posting (dict): must have title, company, description.
        profile (dict | None): defaults to load_profile().
        use_rag (bool): False dumps the whole CV into the prompt instead of
            retrieving chunks - kept only as a Phase 8 groundedness baseline.

    Returns:
        dict: {"profile": {"summary", "highlights", "skills"}, "chunks":
        list[dict], "prompt": str} - chunks is empty when use_rag is False,
        since nothing was retrieved; prompt is the exact text sent to Gemini
        (Phase 8.3 measures its token count to compare RAG's retrieved-chunk
        prompt against the no-RAG full-CV dump it's competing against).
    """
    profile = profile or load_profile()
    llm = GoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=os.environ["GEMINI_API_KEY"])
    common = {
        "name": profile.get("title", "the candidate"),
        "title": posting.get("title", ""),
        "company": posting.get("company", ""),
        "job_description": posting.get("description", ""),
    }

    if use_rag:
        chunks = retrieve_cv_chunks(posting, top_k=top_k)
        bullets = "\n".join(f"- {c['text']}" for c in chunks) or "(no matching experience found)"
        prompt = _PROMPT_TEMPLATE.format(bullets=bullets, schema=_SCHEMA_EXAMPLE, **common)
    else:
        chunks = []
        prompt = _NO_RAG_PROMPT_TEMPLATE.format(full_cv=_flatten_cv(profile), schema=_SCHEMA_EXAMPLE, **common)

    logger.info("Drafting CV profile for %r at %r (rag=%s)", common["title"], common["company"], use_rag)
    raw = llm.invoke(prompt)
    tailored_profile = _parse_profile_json(raw)
    return {"profile": tailored_profile, "chunks": chunks, "prompt": prompt}


def render_cv_profile(tailored_profile: dict, posting: dict, candidate_name: str) -> str:
    """Format a tailored CV profile as a submission-ready Markdown document."""
    lines = [
        f"# {candidate_name} — {posting.get('title', '')} @ {posting.get('company', '')}",
        "",
        "## Summary",
        tailored_profile.get("summary", ""),
        "",
        "## Highlights",
    ]
    lines.extend(f"- {h}" for h in tailored_profile.get("highlights", []))
    lines += ["", "## Skills", ", ".join(tailored_profile.get("skills", []))]
    return "\n".join(lines)


def save_cv_profile(tailored_profile: dict, posting: dict, candidate_name: str) -> Path:
    """Render a tailored CV profile and save it to applications/{company}_{title}.md.

    Returns:
        Path: where the rendered profile was written.
    """
    _APPLICATIONS_DIR.mkdir(exist_ok=True, parents=True)
    company = re.sub(r"[^\w-]+", "_", posting.get("company", "")).strip("_")
    title = re.sub(r"[^\w-]+", "_", posting.get("title", "")).strip("_")
    path = _APPLICATIONS_DIR / f"{company}_{title}.md"
    path.write_text(render_cv_profile(tailored_profile, posting, candidate_name), encoding="utf-8")
    logger.info("Saved tailored CV profile to %s", path)
    return path


def _smoke_test():
    profile = load_profile()
    posting = {
        "title": "Full Stack Engineer",
        "company": "Example Corp",
        "description": "We need someone with React, Node.js and PostgreSQL experience, "
                        "who has built microservices and worked with event-driven systems.",
    }
    result = draft_cv_profile(posting, profile)
    console.print(f"[bold]CV profile[/bold] ({len(result['chunks'])} chunk(s) used):\n")
    console.print(result["profile"])
    console.print("\n[bold]Evidence:[/bold]")
    for c in result["chunks"]:
        console.print(f"  [{c['similarity']:.3f}] {c['text']}")

    _SMOKE_TEST_OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    _SMOKE_TEST_OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"\n[green]Saved result to {_SMOKE_TEST_OUTPUT_PATH}[/green]")


if __name__ == "__main__":
    _smoke_test()
