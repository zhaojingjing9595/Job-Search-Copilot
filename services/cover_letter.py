"""
cover_letter.py

description: RAG cover-letter drafting (Phase 4.5). Phases 2.5/2.6 are vector
*search* - nothing retrieved conditions a generation. This is what makes
"RAG" accurate: the JD is embedded, the top-k `cv_chunks` are retrieved from
the vector store (services/vector_store.py), and only those chunks are
injected into the Gemini prompt. The letter is returned alongside the chunks
that produced it, so every draft ships with its evidence and nothing in it
should trace back to anything else.

A no-RAG variant (whole CV dumped into the prompt) is kept behind a flag
purely so Phase 8 groundedness evaluation has a baseline to compare against.

usage: python -m services.cover_letter   # smoke test against whatever
                                          # posting/profile is already indexed
"""
import os

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

_PROMPT_TEMPLATE = """You are drafting a cover letter for {name}, applying to the {title} role at {company}.

Job description:
{job_description}

Use ONLY the experience bullets below - do not invent experience, employers, or skills that aren't in them. \
If the posting asks for something none of these bullets cover, say so explicitly in the letter rather than \
papering over the gap.

Experience bullets:
{bullets}

Write a concise, specific cover letter (3-4 short paragraphs) grounded entirely in the bullets above."""

_NO_RAG_PROMPT_TEMPLATE = """You are drafting a cover letter for {name}, applying to the {title} role at {company}.

Job description:
{job_description}

Full candidate background:
{full_cv}

Write a concise, specific cover letter (3-4 short paragraphs)."""


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


def draft_cover_letter(posting: dict, profile: dict | None = None, top_k: int = DEFAULT_TOP_K,
                        use_rag: bool = True) -> dict:
    """Draft a cover letter for one posting.

    Args:
        posting (dict): must have title, company, description.
        profile (dict | None): defaults to load_profile().
        use_rag (bool): False dumps the whole CV into the prompt instead of
            retrieving chunks - kept only as a Phase 8 groundedness baseline.

    Returns:
        dict: {"letter": str, "chunks": list[dict], "prompt": str} - chunks
        is empty when use_rag is False, since nothing was retrieved; prompt
        is the exact text sent to Gemini (Phase 8.3 measures its token count
        to compare RAG's retrieved-chunk prompt against the no-RAG full-CV
        dump it's competing against).
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
        prompt = _PROMPT_TEMPLATE.format(bullets=bullets, **common)
    else:
        chunks = []
        prompt = _NO_RAG_PROMPT_TEMPLATE.format(full_cv=_flatten_cv(profile), **common)

    logger.info("Drafting cover letter for %r at %r (rag=%s)", common["title"], common["company"], use_rag)
    letter = llm.invoke(prompt)
    return {"letter": letter.strip(), "chunks": chunks, "prompt": prompt}


def _smoke_test():
    profile = load_profile()
    posting = {
        "title": "Full Stack Engineer",
        "company": "Example Corp",
        "description": "We need someone with React, Node.js and PostgreSQL experience, "
                        "who has built microservices and worked with event-driven systems.",
    }
    result = draft_cover_letter(posting, profile)
    console.print(f"[bold]Cover letter[/bold] ({len(result['chunks'])} chunk(s) used):\n")
    console.print(result["letter"])
    console.print("\n[bold]Evidence:[/bold]")
    for c in result["chunks"]:
        console.print(f"  [{c['similarity']:.3f}] {c['text']}")


if __name__ == "__main__":
    _smoke_test()
