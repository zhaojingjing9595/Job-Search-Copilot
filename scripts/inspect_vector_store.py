"""
inspect_vector_store.py

description: Read-only dump of what's actually stored in the Chroma vector
store (./vector_store) - prints cv_chunks and postings as tables, so you can
sanity-check indexing without writing ad-hoc queries each time.

usage: python -m scripts.inspect_vector_store
"""
from rich.console import Console
from rich.table import Table

from services.vector_store import CV_CHUNKS_COLLECTION, POSTINGS_COLLECTION, _get_collection

console = Console()


def _dump_cv_chunks() -> None:
    collection = _get_collection(CV_CHUNKS_COLLECTION)
    data = collection.get(include=["documents", "metadatas"])
    table = Table(title=f"cv_chunks ({len(data['ids'])} chunk(s))")
    table.add_column("id", overflow="fold")
    table.add_column("company")
    table.add_column("role")
    table.add_column("date_range")
    table.add_column("text", overflow="fold")
    for i, chunk_id in enumerate(data["ids"]):
        meta = data["metadatas"][i]
        table.add_row(chunk_id, meta.get("company", ""), meta.get("role", ""), meta.get("date_range", ""), data["documents"][i])
    console.print(table)


def _dump_postings() -> None:
    collection = _get_collection(POSTINGS_COLLECTION)
    data = collection.get(include=["documents", "metadatas"])
    table = Table(title=f"postings ({len(data['ids'])} posting(s))")
    table.add_column("id", overflow="fold")
    table.add_column("title")
    table.add_column("company")
    table.add_column("location")
    table.add_column("metadata keys", overflow="fold")
    for i, posting_id in enumerate(data["ids"]):
        meta = data["metadatas"][i]
        table.add_row(
            posting_id,
            str(meta.get("title", "")),
            str(meta.get("company", "")),
            str(meta.get("location", "")),
            ", ".join(sorted(meta.keys())),
        )
    console.print(table)


if __name__ == "__main__":
    _dump_cv_chunks()
    console.print()
    _dump_postings()
