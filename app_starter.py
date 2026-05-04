"""
FastAPI skeleton for the banking GenAI workshop (Hard track).

TODO endpoints are stubbed; listings, static files, and helpers are complete.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.cursor import Cursor
import voyageai

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

MONGODB_URI = os.getenv("MONGODB_URI", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
LLM_API_URL = os.getenv("LLM_API_URL", "")

VOYAGE_EMBED_MODEL = "voyage-4-large"

FAQ_VECTOR_INDEX = "faq_vector_index"
KYC_VECTOR_INDEX = "kyc_vector_index"
PRODUCT_VECTOR_INDEX = "product_vector_index"
EMBEDDING_FIELD = "embedding"


def _serialize(obj: Any) -> Any:
    """Convert BSON-friendly structures for JSON: stringify _id, round floats."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "_id" and isinstance(v, ObjectId):
                out["_id"] = str(v)
            else:
                out[k] = _serialize(v)
        return out
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def _serialize_cursor(cursor: Cursor[Any]) -> List[dict[str, Any]]:
    return [_serialize(doc) for doc in cursor]


def parse_object_id(value: str, name: str = "id") -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {name}") from exc


def get_query_embedding(voyage_client: voyageai.Client, text: str) -> List[float]:
    """Embed a search query with Voyage (query-optimized)."""
    response = voyage_client.embed(
        texts=[text],
        model=VOYAGE_EMBED_MODEL,
        input_type="query",
    )
    if not response.embeddings:
        raise HTTPException(status_code=502, detail="Embedding provider returned no vectors")
    return list(response.embeddings[0])


def compute_risk_score(customer: dict[str, Any], loan: dict[str, Any]) -> tuple[int, str, str, dict[str, Any]]:
    """
    Deterministic risk score from customer + loan fields.
    Returns (score 0-100, risk_level, decision, factors).
    """
    score = 50
    factors: dict[str, Any] = {}

    credit_score = int(customer.get("credit_score") or 0)
    factors["credit_score"] = credit_score
    if credit_score > 700:
        score += 20
    elif credit_score > 600:
        score += 10
    elif credit_score > 0 and credit_score < 400:
        score -= 20

    payment_history = str(customer.get("payment_history") or "").strip().lower()
    factors["payment_history"] = payment_history or None
    ph_map = {"excellent": 15, "good": 10, "fair": 0, "poor": -15}
    if payment_history in ph_map:
        score += ph_map[payment_history]

    monthly_income = float(customer.get("monthly_income") or 0)
    monthly_payment = float(loan.get("monthly_payment") or 0)
    dti = (monthly_payment / monthly_income * 100.0) if monthly_income > 0 else 0.0
    factors["dti"] = round(dti, 2)
    if monthly_income <= 0:
        factors["dti_note"] = "no_income_stated"
    if dti < 30:
        score += 10
    elif dti < 50:
        score += 0
    else:
        score -= 15

    account_age = int(customer.get("account_age_months") or 0)
    factors["account_age"] = account_age
    if account_age > 36:
        score += 5

    employment = str(customer.get("employment_type") or "").strip().lower()
    factors["employment"] = employment or None
    if employment in ("salaried", "government"):
        score += 5

    score = max(0, min(100, score))

    if score >= 65:
        risk_level = "low"
    elif score >= 40:
        risk_level = "medium"
    else:
        risk_level = "high"

    decision = "approved" if risk_level != "high" else "declined"

    return score, risk_level, decision, factors


def _credit_explanation_prompt(
    customer: dict[str, Any],
    loan: dict[str, Any],
    risk_score: int,
    risk_level: str,
    decision: str,
    factors: dict[str, Any],
) -> str:
    return (
        "You explain retail loan underwriting decisions clearly for bank staff.\n"
        "Summarize why this application received the given decision, referencing the factors below.\n"
        "Be concise (3-5 sentences), no markdown headings.\n\n"
        f"Risk score: {risk_score}/100\n"
        f"Risk level: {risk_level}\n"
        f"Decision: {decision}\n"
        f"Factors (JSON): {factors}\n"
        f"Customer (subset): { {k: customer.get(k) for k in ('full_name', 'credit_score', 'monthly_income', 'employment_type', 'payment_history', 'account_age_months')} }\n"
        f"Loan (subset): { {k: loan.get(k) for k in ('loan_amount', 'monthly_payment', 'purpose', 'term_months')} }\n"
    )


def _build_customer_profile_text(customer: dict[str, Any]) -> str:
    parts = [
        f"Name: {customer.get('full_name', '')}",
        f"Monthly income: {customer.get('monthly_income', '')}",
        f"Employment: {customer.get('employment_type', '')}",
        f"Payment history: {customer.get('payment_history', '')}",
        f"Credit score: {customer.get('credit_score', '')}",
        f"Goals / notes: {customer.get('financial_goals', '') or customer.get('notes', '')}",
    ]
    return "\n".join(str(p) for p in parts if p.split(": ", 1)[-1])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not set")
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set")
    if not LLM_API_URL:
        raise RuntimeError("LLM_API_URL is not set")

    client = MongoClient(MONGODB_URI)
    db = client["banking"]
    faq_chunks: Collection[Any] = db["faq_chunks"]
    customers: Collection[Any] = db["customers"]
    loan_applications: Collection[Any] = db["loan_applications"]
    kyc_documents: Collection[Any] = db["kyc_documents"]
    bank_products: Collection[Any] = db["bank_products"]
    accounts: Collection[Any] = db["accounts"]
    transactions: Collection[Any] = db["transactions"]

    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    app.state.mongo_client = client
    app.state.db = db
    app.state.faq_chunks = faq_chunks
    app.state.customers = customers
    app.state.loan_applications = loan_applications
    app.state.kyc_documents = kyc_documents
    app.state.bank_products = bank_products
    app.state.accounts = accounts
    app.state.transactions = transactions
    app.state.voyage = voyage_client
    app.state.llm_url = LLM_API_URL

    yield

    client.close()


app = FastAPI(title="GenAI for Financial Services Workshop (Starter)", lifespan=lifespan)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found in static/")
    return FileResponse(index_path)


@app.get("/api/faq")
async def api_faq(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=25),
) -> dict[str, Any]:
    # TODO 1: implement vector search on faq_chunks (FAQ_VECTOR_INDEX) + RAG via LLM_API_URL.
    _ = limit
    return {"query": q, "answer": "", "sources": []}


@app.get("/api/credit-score")
async def api_credit_score(
    customer_id: str = Query(...),
    loan_id: str = Query(...),
) -> dict[str, Any]:
    # TODO 2: implement $lookup aggregation on customers + loan_applications, compute_risk_score, LLM explanation via LLM_API_URL.
    _ = customer_id, loan_id
    return {
        "customer": {},
        "loan": {},
        "risk_score": 0,
        "risk_level": "high",
        "decision": "declined",
        "explanation": "",
        "factors": {
            "dti": None,
            "payment_history": None,
            "credit_score": None,
            "account_age": None,
            "employment": None,
        },
    }


@app.get("/api/customers")
async def api_customers(request: Request) -> list[dict[str, Any]]:
    coll: Collection[Any] = request.app.state.customers
    return _serialize_cursor(coll.find({}))


@app.get("/api/loan-applications")
async def api_loan_applications(
    request: Request,
    customer_id: Optional[str] = Query(None),
) -> list[dict[str, Any]]:
    coll: Collection[Any] = request.app.state.loan_applications
    query_filter: dict[str, Any] = {}
    if customer_id:
        query_filter["customer_id"] = parse_object_id(customer_id, "customer_id")
    return _serialize_cursor(coll.find(query_filter))


@app.get("/api/kyc-check")
async def api_kyc_check(
    request: Request,
    document_id: str = Query(...),
    threshold: float = Query(0.92, ge=0.0, le=1.0),
) -> dict[str, Any]:
    # TODO 3: load document, embed description, $vectorSearch on kyc_documents (KYC_VECTOR_INDEX), exclude self, apply threshold, expiry + risk_flags.
    _ = threshold
    kyc_coll: Collection[Any] = request.app.state.kyc_documents
    doc_oid = parse_object_id(document_id, "document_id")
    doc = kyc_coll.find_one({"_id": doc_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="KYC document not found")

    doc_out = dict(doc)
    doc_out.pop(EMBEDDING_FIELD, None)

    return {
        "document": _serialize(doc_out),
        "duplicates": [],
        "expired": False,
        "risk_flags": [],
    }


@app.get("/api/kyc-documents")
async def api_kyc_documents(request: Request) -> list[dict[str, Any]]:
    coll: Collection[Any] = request.app.state.kyc_documents
    projection = {EMBEDDING_FIELD: 0}
    return _serialize_cursor(coll.find({}, projection))


@app.get("/api/accounts")
async def api_accounts(
    request: Request,
    customer_id: str = Query(...),
) -> list[dict[str, Any]]:
    coll: Collection[Any] = request.app.state.accounts
    cust_oid = parse_object_id(customer_id, "customer_id")
    return _serialize_cursor(coll.find({"customer_id": cust_oid}))


@app.get("/api/transactions")
async def api_transactions(
    request: Request,
    account_id: Optional[str] = Query(None),
    customer_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200),
) -> list[dict[str, Any]]:
    coll: Collection[Any] = request.app.state.transactions
    query_filter: dict[str, Any] = {}
    if account_id:
        query_filter["account_id"] = parse_object_id(account_id, "account_id")
    elif customer_id:
        query_filter["customer_id"] = parse_object_id(customer_id, "customer_id")
    cursor = coll.find(query_filter).sort("date", -1).limit(limit)
    return _serialize_cursor(cursor)


@app.get("/api/recommend-products")
async def api_recommend_products(
    request: Request,
    customer_id: str = Query(...),
    limit: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    # TODO 4: build profile with _build_customer_profile_text, embed, $vectorSearch on bank_products (PRODUCT_VECTOR_INDEX).
    _ = limit
    customers: Collection[Any] = request.app.state.customers
    cust_oid = parse_object_id(customer_id, "customer_id")
    if customers.find_one({"_id": cust_oid}) is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    return {"customer_id": customer_id, "recommendations": []}
