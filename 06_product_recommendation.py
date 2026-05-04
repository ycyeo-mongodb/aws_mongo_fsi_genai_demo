#!/usr/bin/env python3
"""
Workshop 06: Embed bank products with Voyage AI, vector search recommendations for declined profiles.

Requires an Atlas Vector Search index on banking.bank_products (vector field: embedding).
Default index name: bank_products_vector_index — set BANK_PRODUCTS_VECTOR_INDEX in .env to override.
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from voyageai import Client as VoyageClient

SCRIPT_DIR = Path(__file__).resolve().parent

VOYAGE_MODEL = "voyage-4-large"

BANK_PRODUCTS: list[dict[str, str]] = [
    {
        "product_id": "PROD-PERSONAL",
        "name": "Personal Loan",
        "description": (
            "Personal Loan — flexible terms up to $10,000 for salaried and stable-income customers; "
            "fixed monthly installments and optional early repayment."
        ),
    },
    {
        "product_id": "PROD-MICRO",
        "name": "Micro-Business Loan",
        "description": (
            "Micro-Business Loan — for small business owners and market vendors; working capital "
            "with simplified documentation and relationship-manager support."
        ),
    },
    {
        "product_id": "PROD-AGRI",
        "name": "Agricultural Loan",
        "description": (
            "Agricultural Loan — seasonal repayment aligned to harvest cycles; for crop and livestock "
            "producers with flexible grace periods."
        ),
    },
    {
        "product_id": "PROD-HOUSING",
        "name": "Housing Loan",
        "description": (
            "Housing Loan — long-term financing up to 20 years for home purchase or construction; "
            "competitive rates with property collateral."
        ),
    },
    {
        "product_id": "PROD-EDU",
        "name": "Education Loan",
        "description": (
            "Education Loan — tuition and living expenses for students and families; staged "
            "disbursements tied to enrollment."
        ),
    },
    {
        "product_id": "PROD-SECURED",
        "name": "Secured Savings Loan",
        "description": (
            "Secured Savings Loan — borrow against your fixed deposit or savings balance; "
            "lower rates and fast approval while keeping savings intact."
        ),
    },
    {
        "product_id": "PROD-SME-LINE",
        "name": "SME Credit Line",
        "description": (
            "SME Credit Line — revolving credit for small and medium enterprises; draw and repay "
            "as cash flow allows, ideal for inventory and payroll."
        ),
    },
    {
        "product_id": "PROD-EMERGENCY",
        "name": "Emergency Loan",
        "description": (
            "Emergency Loan — small, quick-disbursement loans for urgent needs; shorter terms and "
            "streamlined approval for existing customers."
        ),
    },
]


def seed_bank_products(db: Database, voyage: VoyageClient) -> int:
    """Embed product descriptions and store in banking.bank_products (replaces collection)."""
    db.bank_products.drop()
    texts = [p["description"] for p in BANK_PRODUCTS]
    emb = voyage.embed(texts=texts, model=VOYAGE_MODEL, input_type="document")
    vectors = emb.embeddings
    docs: list[dict[str, Any]] = []
    for prod, vec in zip(BANK_PRODUCTS, vectors, strict=True):
        docs.append(
            {
                "product_id": prod["product_id"],
                "name": prod["name"],
                "description": prod["description"],
                "embedding": vec,
                "embedding_model": VOYAGE_MODEL,
            }
        )
    if docs:
        db.bank_products.insert_many(docs)
    return len(docs)


def recommend_products(
    db: Database,
    voyage: VoyageClient,
    customer_profile_text: str,
    top_k: int = 3,
    index_name: str | None = None,
) -> list[dict[str, Any]]:
    """Embed profile as query; run $vectorSearch on banking.bank_products."""
    idx = index_name or os.environ.get("BANK_PRODUCTS_VECTOR_INDEX", "bank_products_vector_index")
    q = voyage.embed(
        texts=[customer_profile_text],
        model=VOYAGE_MODEL,
        input_type="query",
    )
    query_vector = q.embeddings[0]

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": idx,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": top_k,
            }
        },
        {"$set": {"similarity_score": {"$meta": "vectorSearchScore"}}},
        {
            "$project": {
                "product_id": 1,
                "name": 1,
                "description": 1,
                "similarity_score": 1,
            }
        },
    ]
    return list(db.bank_products.aggregate(pipeline))


def profile_summary_for_declined(customer: dict[str, Any], loan: dict[str, Any]) -> str:
    """Short natural-language profile for embedding (declined / high-friction scenario)."""
    return (
        f"Customer {customer.get('name')} is a {customer.get('employment_type', 'unknown')} "
        f"with monthly income around ${float(customer.get('income_usd') or 0):,.0f}, "
        f"credit score {customer.get('credit_score')}, payment history {customer.get('payment_history')}, "
        f"account age {customer.get('account_age_months')} months. "
        f"A loan application for {loan.get('purpose')} of ${float(loan.get('amount_usd') or 0):,.0f} "
        f"was not suitable; seek alternative smaller or secured products aligned to cash flow."
    )


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")
    uri = os.environ.get("MONGODB_URI")
    v_key = os.environ.get("VOYAGE_API_KEY")
    if not uri or not v_key:
        raise SystemExit("Set MONGODB_URI and VOYAGE_API_KEY in .env")

    mongo = MongoClient(uri)
    db = mongo["banking"]
    voyage = VoyageClient(api_key=v_key)

    print("Embedding bank products and loading banking.bank_products …")
    n = seed_bank_products(db, voyage)
    print(f"  Loaded {n} products with embeddings ({VOYAGE_MODEL}).")
    print()
    print(
        "Note: Create an Atlas Vector Search index on banking.bank_products\n"
        "  field path: embedding | dimensions: 1024 | similarity: cosine\n"
        "  index name (default): bank_products_vector_index\n"
    )

    declined = list(db.loan_applications.find({"status": "declined"}).limit(3))
    print("=== Recommendations for declined applications (top 3 each) ===\n")

    index_name = os.environ.get("BANK_PRODUCTS_VECTOR_INDEX", "bank_products_vector_index")

    for loan in declined:
        cid = loan.get("customer_id")
        cust = db.customers.find_one({"id": cid})
        if not cust:
            print(f"No customer {cid} for loan {loan.get('id')}\n")
            continue
        text = profile_summary_for_declined(cust, loan)
        print("—" * 60)
        print(f"Customer: {cust.get('name')} ({cid})  |  Declined loan: {loan.get('id')}")
        print(f"Profile text (embedded):\n  {text[:220]}…\n")
        try:
            recs = recommend_products(db, voyage, text, top_k=3, index_name=index_name)
        except Exception as e:
            print(f"Vector search failed (is the index deployed?). Error: {e}\n")
            continue
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r.get('name')} (score={r.get('similarity_score', 0):.4f})")
            print(f"     {r.get('description', '')[:120]}…")
        print()

    mongo.close()
    print("Done.")


if __name__ == "__main__":
    main()
