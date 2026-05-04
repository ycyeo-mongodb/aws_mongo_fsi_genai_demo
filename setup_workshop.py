#!/usr/bin/env python3
"""
Single-command workshop setup: loads all data, embeds with Voyage AI, and creates Atlas indexes.

Combines: 01_load_faq_data.py, 04_load_customers.py, 06_product_recommendation.py,
          07_load_kyc_data.py, 02_create_indexes.py

Usage:
    python setup_workshop.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel
import voyageai

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DB_NAME = "banking"
VOYAGE_MODEL = "voyage-4-large"
EMBED_BATCH_SIZE = 25
CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP = 150


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def load_json(name: str) -> list[dict]:
    path = DATA_DIR / name
    if not path.is_file():
        raise SystemExit(f"Missing data file: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def embed_batch(voyage: voyageai.Client, texts: list[str], input_type: str = "document") -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    total = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for batch_num, i in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), start=1):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        print(f"    Embedding batch {batch_num}/{total} ({len(batch)} text(s)) ...")
        result = voyage.embed(batch, model=VOYAGE_MODEL, input_type=input_type)
        all_embeddings.extend(result.embeddings)
    return all_embeddings


def index_exists(collection, name: str) -> bool:
    try:
        for idx in collection.list_search_indexes():
            if idx.get("name") == name:
                return True
    except PyMongoError:
        pass
    return False


# ──────────────────────────────────────────────────────────
# Step 1: FAQ chunks + embeddings
# ──────────────────────────────────────────────────────────

def build_combined_text(item: dict) -> str:
    parts = [
        f"[EN] {item.get('title_en', '').strip()}",
        item.get("content_en", "").strip(),
        "",
        f"[KM] {item.get('title_km', '').strip()}",
        item.get("content_km", "").strip(),
    ]
    return "\n".join(parts).strip()


def chunk_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_MAX_CHARS:
        return [text]

    segments: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_MAX_CHARS, n)
        if end < n:
            break_at = text.rfind("\n", start, end)
            if break_at <= start:
                break_at = text.rfind(" ", start, end)
            if break_at > start:
                end = break_at
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
        if end >= n:
            break
        start = max(0, end - CHUNK_OVERLAP)
    return segments


def step_load_faq(db, voyage: voyageai.Client) -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Load FAQ data → banking.faq_chunks")
    print("=" * 60)

    policies = load_json("faq_policies.json")
    chunk_docs: list[dict] = []
    for item in policies:
        source_id = item.get("id", "")
        combined = build_combined_text(item)
        segments = chunk_text(combined)
        title_en = (item.get("title_en") or "").strip()
        for chunk_text_val in segments:
            chunk_docs.append(
                {
                    "source_id": source_id,
                    "title": title_en,
                    "content_en": item.get("content_en", ""),
                    "content_km": item.get("content_km", ""),
                    "category": item.get("category", ""),
                    "department": item.get("department", ""),
                    "language": "bilingual",
                    "chunk_text": chunk_text_val,
                }
            )

    print(f"  {len(policies)} policies → {len(chunk_docs)} chunk(s)")
    embeddings = embed_batch(voyage, [d["chunk_text"] for d in chunk_docs])
    for doc, emb in zip(chunk_docs, embeddings):
        doc["embedding"] = emb

    db.faq_chunks.drop()
    if chunk_docs:
        db.faq_chunks.insert_many(chunk_docs)
    print(f"  ✓ Inserted {db.faq_chunks.count_documents({})} faq_chunks")


# ──────────────────────────────────────────────────────────
# Step 2: Customers, accounts, loans, transactions
# ──────────────────────────────────────────────────────────

def step_load_customers(db) -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Load customers, accounts, loans, transactions")
    print("=" * 60)

    customers_raw = load_json("customers.json")
    loans_raw = load_json("loan_applications.json")
    accounts_raw = load_json("accounts.json")
    transactions_raw = load_json("transactions.json")

    for name in ("customers", "loan_applications", "accounts", "transactions"):
        db[name].drop()

    cust_id_map: dict[str, ObjectId] = {}
    for doc in customers_raw:
        oid = ObjectId()
        cust_id_map[doc["id"]] = oid
        doc["_id"] = oid
        doc["full_name"] = doc["name"]
        doc["monthly_income"] = doc.get("income_usd", 0)
    if customers_raw:
        db.customers.insert_many(customers_raw)

    acc_id_map: dict[str, ObjectId] = {}
    for doc in accounts_raw:
        oid = ObjectId()
        acc_id_map[doc["id"]] = oid
        doc["_id"] = oid
        cust_str = doc.get("customer_id", "")
        doc["customer_id"] = cust_id_map.get(cust_str, cust_str)
    if accounts_raw:
        db.accounts.insert_many(accounts_raw)

    for doc in loans_raw:
        oid = ObjectId()
        doc["_id"] = oid
        cust_str = doc.get("customer_id", "")
        doc["customer_id"] = cust_id_map.get(cust_str, cust_str)
        doc["loan_amount"] = doc.get("amount_usd", 0)
        doc["monthly_payment"] = doc.get("monthly_payment_usd", 0)
    if loans_raw:
        db.loan_applications.insert_many(loans_raw)

    for doc in transactions_raw:
        oid = ObjectId()
        doc["_id"] = oid
        cust_str = doc.get("customer_id", "")
        doc["customer_id"] = cust_id_map.get(cust_str, cust_str)
        acc_str = doc.get("account_id", "")
        doc["account_id"] = acc_id_map.get(acc_str, acc_str)
    if transactions_raw:
        db.transactions.insert_many(transactions_raw)

    print(f"  ✓ customers:         {db.customers.count_documents({})}")
    print(f"  ✓ accounts:          {db.accounts.count_documents({})}")
    print(f"  ✓ loan_applications: {db.loan_applications.count_documents({})}")
    print(f"  ✓ transactions:      {db.transactions.count_documents({})}")


# ──────────────────────────────────────────────────────────
# Step 3: Bank products + embeddings
# ──────────────────────────────────────────────────────────

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


def step_load_products(db, voyage: voyageai.Client) -> None:
    print("\n" + "=" * 60)
    print("STEP 3: Load bank products → banking.bank_products")
    print("=" * 60)

    db.bank_products.drop()
    texts = [p["description"] for p in BANK_PRODUCTS]
    print(f"  Embedding {len(texts)} product descriptions ...")
    result = voyage.embed(texts=texts, model=VOYAGE_MODEL, input_type="document")
    vectors = result.embeddings

    docs = []
    for prod, vec in zip(BANK_PRODUCTS, vectors):
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
    print(f"  ✓ Inserted {db.bank_products.count_documents({})} bank_products")


# ──────────────────────────────────────────────────────────
# Step 4: KYC documents + embeddings
# ──────────────────────────────────────────────────────────

def step_load_kyc(db, voyage: voyageai.Client) -> None:
    print("\n" + "=" * 60)
    print("STEP 4: Load KYC documents → banking.kyc_documents")
    print("=" * 60)

    docs = load_json("kyc_documents.json")
    descriptions = [d.get("description") or "" for d in docs]
    embeddings = embed_batch(voyage, descriptions)
    for doc, vec in zip(docs, embeddings):
        doc["description_embedding"] = vec
        doc["embedding_model"] = VOYAGE_MODEL
        b64 = doc.pop("pdf_data_b64", "")
        if b64:
            doc["pdf_data"] = base64.b64decode(b64)

    db.kyc_documents.drop()
    if docs:
        db.kyc_documents.insert_many(docs)
    pdf_count = sum(1 for d in docs if d.get("pdf_data"))
    print(f"  ✓ Inserted {db.kyc_documents.count_documents({})} kyc_documents ({pdf_count} with PDF scans)")


def step_load_documents(db) -> None:
    """Load pre-generated PDF documents (payslips + bank statements) from data/documents.json."""
    print("\n" + "=" * 60)
    print("STEP 5: Load PDF documents → banking.documents")
    print("=" * 60)

    raw_docs = load_json("documents.json")
    cust_id_to_oid = {
        doc.get("id"): doc["_id"]
        for doc in db.customers.find({}, {"id": 1})
    }

    docs_to_insert = []
    for d in raw_docs:
        cust_str = d.get("customer_id", "")
        cust_oid = cust_id_to_oid.get(cust_str)
        docs_to_insert.append({
            "customer_id": cust_oid,
            "customer_id_str": cust_str,
            "doc_type": d["doc_type"],
            "filename": d["filename"],
            "pdf_data": base64.b64decode(d["pdf_data_b64"]),
            "text_content": d.get("text_content", ""),
            "generated_date": d.get("generated_date", ""),
        })

    db.documents.drop()
    if docs_to_insert:
        db.documents.insert_many(docs_to_insert)
    print(f"  ✓ Inserted {len(docs_to_insert)} PDF documents (payslips + bank statements)")


# ──────────────────────────────────────────────────────────
# Step 6: Create Atlas Search / Vector Search indexes
# ──────────────────────────────────────────────────────────

def step_create_indexes(db) -> None:
    print("\n" + "=" * 60)
    print("STEP 6: Create Atlas Vector Search indexes")
    print("=" * 60)

    indexes = [
        (
            db["faq_chunks"],
            "faq_vector_index",
            {
                "fields": [
                    {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
                    {"type": "filter", "path": "category"},
                    {"type": "filter", "path": "language"},
                ]
            },
        ),
        (
            db["kyc_documents"],
            "kyc_vector_index",
            {
                "fields": [
                    {"type": "vector", "path": "description_embedding", "numDimensions": 1024, "similarity": "cosine"},
                ]
            },
        ),
        (
            db["bank_products"],
            "product_vector_index",
            {
                "fields": [
                    {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
                ]
            },
        ),
    ]

    for col, name, definition in indexes:
        if index_exists(col, name):
            print(f"\n  [{col.name}] {name} — already exists, skipping")
        else:
            print(f"\n  [{col.name}] {name} — creating ...")
            col.create_search_index(model=SearchIndexModel(
                definition=definition,
                name=name,
                type="vectorSearch",
            ))
            print(f"    ✓ Requested {name}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")

    mongodb_uri = require_env("MONGODB_URI")
    voyage_key = require_env("VOYAGE_API_KEY")

    print("Connecting to MongoDB Atlas ...")
    try:
        mongo = MongoClient(mongodb_uri, serverSelectionTimeoutMS=15000)
        mongo.admin.command("ping")
    except PyMongoError as e:
        print(f"ERROR: Could not connect to MongoDB: {e}", file=sys.stderr)
        sys.exit(1)
    print("  Connected.\n")

    db = mongo[DB_NAME]
    voyage = voyageai.Client(api_key=voyage_key)

    t0 = time.time()

    step_load_faq(db, voyage)
    step_load_customers(db)
    step_load_products(db, voyage)
    step_load_kyc(db, voyage)
    step_load_documents(db)
    step_create_indexes(db)

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
    print(f"  Database: {DB_NAME}")
    print(f"  Collections loaded: faq_chunks, customers, accounts,")
    print(f"    loan_applications, transactions, bank_products,")
    print(f"    kyc_documents, documents")
    print(f"  PDF documents: ~100 (payslips + bank statements)")
    print(f"  Indexes requested: faq_vector_index, kyc_vector_index,")
    print(f"    product_vector_index  (3 total — M0 free tier limit)")
    print(f"  Total time: {elapsed:.1f}s")
    print()
    print("  ⚠  Atlas builds search indexes asynchronously.")
    print("     Wait until all indexes show ACTIVE in the Atlas UI")
    print("     before testing vector search queries in the app.")

    mongo.close()


if __name__ == "__main__":
    main()
