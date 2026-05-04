#!/usr/bin/env python3
"""
Workshop 08: KYC verification — expiry/missing fields, duplicate detection via $vectorSearch.

Uses index name kyc_vector_index on path description_embedding (see 07_load_kyc_data.py).
Override with KYC_VECTOR_INDEX in .env if needed.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from voyageai import Client as VoyageClient

SCRIPT_DIR = Path(__file__).resolve().parent

VOYAGE_MODEL = "voyage-4-large"
REQUIRED_FIELDS = (
    "id",
    "customer_id",
    "document_type",
    "document_number",
    "issue_date",
    "expiry_date",
    "description",
)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def check_duplicate(
    db: Database,
    voyage: VoyageClient,
    doc_description: str,
    threshold: float = 0.92,
    exclude_id: str | None = None,
    index_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Embed description and vector-search kyc_documents; return matches at or above similarity threshold.
    """
    idx = index_name or os.environ.get("KYC_VECTOR_INDEX", "kyc_vector_index")
    q = voyage.embed(
        texts=[doc_description],
        model=VOYAGE_MODEL,
        input_type="query",
    )
    query_vector = q.embeddings[0]

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": idx,
                "path": "description_embedding",
                "queryVector": query_vector,
                "numCandidates": 50,
                "limit": 10,
            }
        },
        {"$set": {"similarity_score": {"$meta": "vectorSearchScore"}}},
    ]
    if exclude_id:
        pipeline.append({"$match": {"id": {"$ne": exclude_id}}})

    pipeline.append(
        {
            "$project": {
                "id": 1,
                "customer_id": 1,
                "document_type": 1,
                "document_number": 1,
                "description": 1,
                "similarity_score": 1,
            }
        }
    )

    raw = list(db.kyc_documents.aggregate(pipeline))
    return [m for m in raw if float(m.get("similarity_score") or 0) >= threshold]


def verify_document(
    db: Database,
    voyage: VoyageClient,
    kyc_doc: dict[str, Any],
    duplicate_threshold: float = 0.92,
    index_name: str | None = None,
) -> dict[str, Any]:
    """Check expiry, missing fields, then semantic duplicate search."""
    findings: list[str] = []
    today = date.today()

    missing = [f for f in REQUIRED_FIELDS if not kyc_doc.get(f)]
    if missing:
        findings.append(f"missing_fields:{','.join(missing)}")

    exp = _parse_date(kyc_doc.get("expiry_date"))
    if exp is None:
        findings.append("invalid_or_missing_expiry_date")
    elif exp < today:
        findings.append("document_expired")

    desc = (kyc_doc.get("description") or "").strip()
    dupes: list[dict[str, Any]] = []
    if desc:
        dupes = check_duplicate(
            db,
            voyage,
            desc,
            threshold=duplicate_threshold,
            exclude_id=kyc_doc.get("id"),
            index_name=index_name,
        )
        if dupes:
            findings.append("possible_duplicate_submission")

    return {
        "document_id": kyc_doc.get("id"),
        "customer_id": kyc_doc.get("customer_id"),
        "findings": findings,
        "duplicate_matches": dupes,
    }


def flag_suspicious(verification_results: list[dict[str, Any]]) -> list[str]:
    """Aggregate risk flags from one or more verify_document results."""
    flags: list[str] = []
    for vr in verification_results:
        doc_id = vr.get("document_id")
        for f in vr.get("findings") or []:
            flags.append(f"{doc_id}:{f}")
        for m in vr.get("duplicate_matches") or []:
            mid = m.get("id")
            sc = m.get("similarity_score")
            flags.append(f"{doc_id}:duplicate_candidate_match:{mid}:score={sc:.4f}")
    return flags


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")
    uri = os.environ.get("MONGODB_URI")
    v_key = os.environ.get("VOYAGE_API_KEY")
    if not uri or not v_key:
        raise SystemExit("Set MONGODB_URI and VOYAGE_API_KEY in .env")

    mongo = MongoClient(uri)
    db = mongo["banking"]
    voyage = VoyageClient(api_key=v_key)
    index_name = os.environ.get("KYC_VECTOR_INDEX", "kyc_vector_index")

    # Demo set: expired doc, duplicate pair (KYC-00005 / KYC-00006 share description), clean doc
    demo_ids = ["KYC-00008", "KYC-00006", "KYC-00001"]
    docs = list(db.kyc_documents.find({"id": {"$in": demo_ids}}))
    by_id = {d["id"]: d for d in docs}
    ordered = [by_id[i] for i in demo_ids if i in by_id]

    print("=== KYC verification demo ===\n")
    print(f"Vector index: {index_name} | path: description_embedding\n")

    results: list[dict[str, Any]] = []
    for doc in ordered:
        print("—" * 60)
        print(
            f"Document: {doc.get('id')} | type: {doc.get('document_type')} | "
            f"customer: {doc.get('customer_id')}"
        )
        print(f"Expiry: {doc.get('expiry_date')} | number: {doc.get('document_number')}")
        print(f"Description: {(doc.get('description') or '')[:100]}…")
        try:
            vr = verify_document(db, voyage, doc, index_name=index_name)
        except Exception as e:
            print(f"Verification error (is vector index deployed?): {e}\n")
            continue
        results.append(vr)
        print()
        print("  Findings:", vr.get("findings") or ["(none)"])
        print("  Duplicate matches (score >= 0.92):")
        for m in vr.get("duplicate_matches") or []:
            print(f"    - {m.get('id')}  score={m.get('similarity_score', 0):.4f}")
            print(f"      { (m.get('description') or '')[:90]}…")
        if not vr.get("duplicate_matches"):
            print("    (none)")
        print()

    all_flags = flag_suspicious(results)
    print("=" * 60)
    print("Aggregated risk flags:")
    for fl in all_flags:
        print(f"  • {fl}")
    print()
    print("Done.")

    mongo.close()


if __name__ == "__main__":
    main()
