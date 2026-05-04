#!/usr/bin/env python3
"""
Demo RAG FAQ chatbot: Voyage query embeddings + Atlas Vector Search + Amazon Bedrock (via API Gateway).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import requests
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import voyageai

REPO_ROOT = Path(__file__).resolve().parent
DB_NAME = "banking"
COLLECTION_NAME = "faq_chunks"
VECTOR_INDEX = "faq_vector_index"
VOYAGE_MODEL = "voyage-4-large"

SYSTEM_PROMPT = (
    "You are a helpful banking assistant. "
    "Answer based ONLY on the provided context. "
    "If the context doesn't contain the answer, say so. "
    "Respond in the same language as the question."
)


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def embed_query(voyage_client: voyageai.Client, query: str) -> list[float]:
    try:
        result = voyage_client.embed(
            [query],
            model=VOYAGE_MODEL,
            input_type="query",
        )
    except Exception as e:
        raise RuntimeError(f"Voyage AI embed failed: {e}") from e
    if not result.embeddings:
        raise RuntimeError("Voyage AI returned no embeddings.")
    return result.embeddings[0]


def retrieve_context(
    collection,
    query_embedding: list[float],
    *,
    limit: int = 5,
) -> list[dict]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": min(200, max(limit * 20, 50)),
                "limit": limit,
            }
        },
        {
            "$project": {
                "source_id": 1,
                "title": 1,
                "chunk_text": 1,
                "category": 1,
                "language": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    try:
        return list(collection.aggregate(pipeline))
    except PyMongoError as e:
        raise RuntimeError(
            f"Vector search failed (is '{VECTOR_INDEX}' READY on {DB_NAME}.{COLLECTION_NAME}?): {e}"
        ) from e


def format_context_for_llm(chunks: list[dict]) -> str:
    parts: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        header = f"[{i}] source_id={ch.get('source_id', '')} title={ch.get('title', '')}"
        parts.append(f"{header}\n{ch.get('chunk_text', '')}")
    return "\n\n".join(parts)


def generate_answer(llm_url: str, query: str, context_chunks: list[dict]) -> str:
    context_block = format_context_for_llm(context_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
    ]
    try:
        resp = requests.post(llm_url, json={"messages": messages}, timeout=55)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {e}") from e
    if "error" in data:
        raise RuntimeError(f"LLM API error: {data['error']}")
    return (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()


def print_chunk_line(chunk: dict) -> None:
    score = chunk.get("score")
    score_s = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
    title = (chunk.get("title") or "")[:80]
    text_preview = (chunk.get("chunk_text") or "")[:200].replace("\n", " ")
    print(f"    score={score_s} | {chunk.get('source_id', '')} | {title}")
    print(f"      {text_preview}...")


def run_demo(
    voyage_client: voyageai.Client,
    llm_url: str,
    collection,
    query: str,
    *,
    limit: int = 5,
) -> None:
    print(f"\nQuery: {query}")
    print("-" * 72)
    try:
        q_emb = embed_query(voyage_client, query)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return

    try:
        chunks = retrieve_context(collection, q_emb, limit=limit)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return

    print(f"Retrieved {len(chunks)} chunk(s):")
    for ch in chunks:
        print_chunk_line(ch)

    try:
        answer = generate_answer(llm_url, query, chunks)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return

    print(f"\nAnswer:\n{answer}")


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    mongodb_uri = require_env("MONGODB_URI")
    voyage_key = require_env("VOYAGE_API_KEY")
    llm_url = require_env("LLM_API_URL")

    print("Initializing clients ...")
    voyage_client = voyageai.Client(api_key=voyage_key)

    print("Connecting to MongoDB Atlas ...")
    try:
        mongo = MongoClient(mongodb_uri, serverSelectionTimeoutMS=15000)
        mongo.admin.command("ping")
    except PyMongoError as e:
        print(f"ERROR: Could not connect to MongoDB: {e}", file=sys.stderr)
        sys.exit(1)

    collection = mongo[DB_NAME][COLLECTION_NAME]

    demo_queries = [
        "What is the interest rate for personal loans?",
        "តើការប្រាក់ប្រចាំឆ្នាំសម្រាប់ប្រាក់កម្ចីផ្ទាល់ខ្លួនគឺប៉ុន្មាន?",
        "What documents do I need to open an account?",
        "What is the staff annual leave policy?",
    ]

    print("\n=== Banking FAQ RAG demo ===\n")
    for q in demo_queries:
        run_demo(voyage_client, llm_url, collection, q, limit=5)

    mongo.close()
    print("\n" + "=" * 72)
    print("Demo complete.")


if __name__ == "__main__":
    main()
