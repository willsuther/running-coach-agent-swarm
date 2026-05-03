"""
tools/memory_retrieval.py

Retrieval wrapper for the ChromaDB vector store.
All agents call these functions — no agent queries ChromaDB directly.
"""

from __future__ import annotations
import chromadb
from chromadb.utils import embedding_functions


def get_client(chroma_dir: str, api_key: str) -> tuple:
    client = chromadb.PersistentClient(path=chroma_dir)
    ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name="models/gemini-embedding-001",
    )
    return client, ef


def retrieve_workouts(client, ef, query: str, n: int = 5, filters: dict | None = None) -> list[dict]:
    try:
        col = client.get_collection("workout_summaries", embedding_function=ef)
    except Exception:
        return []
    kwargs = {"query_texts": [query], "n_results": min(n, col.count())}
    if filters:
        kwargs["where"] = filters
    results = col.query(**kwargs)
    return [
        {"document": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def retrieve_profile(client, ef, query: str = "athlete background goals training context", n: int = 5) -> str:
    try:
        col = client.get_collection("athlete_profile", embedding_function=ef)
    except Exception:
        return ""
    results = col.query(query_texts=[query], n_results=min(n, col.count()))
    return "\n\n".join(results["documents"][0])


def retrieve_notes(client, ef, query: str, n: int = 3) -> list[dict]:
    try:
        col = client.get_collection("session_notes", embedding_function=ef)
        if col.count() == 0:
            return []
    except Exception:
        return []
    results = col.query(query_texts=[query], n_results=min(n, col.count()))
    return [
        {"document": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def retrieve_all(client, ef, query: str, n_workouts: int = 5, n_profile: int = 4, n_notes: int = 3) -> dict:
    return {
        "workouts": retrieve_workouts(client, ef, query, n=n_workouts),
        "profile":  retrieve_profile(client, ef, query, n=n_profile),
        "notes":    retrieve_notes(client, ef, query, n=n_notes),
    }


def format_memory_context(results: dict) -> str:
    lines = ["=== Memory Context ==="]
    if results.get("profile"):
        lines += ["\n--- Athlete Profile ---", results["profile"]]
    if results.get("workouts"):
        lines.append("\n--- Relevant Past Workouts ---")
        for r in results["workouts"]:
            lines.append(r["document"])
            lines.append("")
    if results.get("notes"):
        lines.append("--- Session Notes ---")
        for r in results["notes"]:
            d = r["metadata"].get("date", "unknown")
            lines.append(f"[{d}] {r['document']}")
    return "\n".join(lines)
