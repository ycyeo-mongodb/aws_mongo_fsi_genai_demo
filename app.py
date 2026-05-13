"""
Full FastAPI application for the banking GenAI workshop.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

import base64

from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
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
KYC_EMBEDDING_FIELD = "description_embedding"
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


def call_llm(messages: List[dict[str, str]], llm_url: str, temperature: float = 0.2) -> str:
    """Send chat messages to the LLM API Gateway and return the assistant response."""
    payload = {"messages": messages}
    try:
        resp = requests.post(llm_url, json=payload, timeout=55)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LLM API request failed: {exc}") from exc
    data = resp.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=f"LLM API error: {data['error']}")
    return (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()


def call_llm_debug(messages: List[dict[str, str]], llm_url: str) -> dict[str, Any]:
    """Like call_llm but returns the full payload for pipeline transparency."""
    payload = {"messages": messages}
    t0 = time.time()
    try:
        resp = requests.post(llm_url, json=payload, timeout=55)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LLM API request failed: {exc}") from exc
    latency_ms = round((time.time() - t0) * 1000)
    data = resp.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=f"LLM API error: {data['error']}")
    answer = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    return {
        "answer": answer,
        "model": data.get("model", ""),
        "usage": data.get("usage", {}),
        "latency_ms": latency_ms,
    }


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
    documents: Collection[Any] = db["documents"]
    network_graph: Collection[Any] = db["network_graph"]
    insurance_policies: Collection[Any] = db["insurance_policies"]

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
    app.state.documents = documents
    app.state.network_graph = network_graph
    app.state.insurance_policies = insurance_policies
    app.state.voyage = voyage_client
    app.state.llm_url = LLM_API_URL

    yield

    client.close()


app = FastAPI(title="GenAI for Financial Services Workshop", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=25),
) -> dict[str, Any]:
    voyage_client: voyageai.Client = request.app.state.voyage
    llm_url: str = request.app.state.llm_url
    faq_coll: Collection[Any] = request.app.state.faq_chunks

    t_embed_start = time.time()
    query_vector = get_query_embedding(voyage_client, q)
    embed_ms = round((time.time() - t_embed_start) * 1000)

    pipeline: List[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": FAQ_VECTOR_INDEX,
                "path": EMBEDDING_FIELD,
                "queryVector": query_vector,
                "numCandidates": min(200, max(50, limit * 20)),
                "limit": limit,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {EMBEDDING_FIELD: 0}},
    ]

    t_search_start = time.time()
    raw_chunks = list(faq_coll.aggregate(pipeline))
    search_ms = round((time.time() - t_search_start) * 1000)

    sources: List[dict[str, Any]] = []
    context_blocks: List[str] = []
    for doc in raw_chunks:
        score = doc.get("score")
        title = doc.get("title", "")
        content_en = doc.get("content_en", "")
        content_km = doc.get("content_km", "")
        category = doc.get("category", "")
        sources.append(
            {
                "title": title,
                "content_en": content_en,
                "content_km": content_km,
                "category": category,
                "score": round(float(score), 6) if isinstance(score, (int, float)) else score,
            }
        )
        block = f"Title: {title}\nCategory: {category}\nEN: {content_en}\nKM: {content_km}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    system_prompt = (
        "You are a helpful banking assistant. Answer based ONLY on the "
        "provided context. If the context doesn't contain the answer, say so. Respond in the "
        "same language as the question."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{q}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    llm_result = call_llm_debug(messages, llm_url)

    pipeline_readable = [
        {"$vectorSearch": {"index": FAQ_VECTOR_INDEX, "path": EMBEDDING_FIELD, "queryVector": f"<{len(query_vector)}-dim float array>", "numCandidates": min(200, max(50, limit * 20)), "limit": limit}},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {EMBEDDING_FIELD: 0}},
    ]

    return {
        "query": q,
        "answer": llm_result["answer"],
        "sources": sources,
        "debug": {
            "embedding": {
                "model": VOYAGE_EMBED_MODEL,
                "dimensions": len(query_vector),
                "input_type": "query",
                "latency_ms": embed_ms,
            },
            "vector_search": {
                "index": FAQ_VECTOR_INDEX,
                "collection": "banking.faq_chunks",
                "pipeline": pipeline_readable,
                "results_returned": len(raw_chunks),
                "latency_ms": search_ms,
            },
            "llm": {
                "endpoint": llm_url,
                "model": llm_result["model"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "usage": llm_result["usage"],
                "latency_ms": llm_result["latency_ms"],
            },
        },
    }


@app.get("/api/credit-score")
async def api_credit_score(
    request: Request,
    customer_id: str = Query(...),
    loan_id: str = Query(...),
) -> dict[str, Any]:
    customers: Collection[Any] = request.app.state.customers
    loans: Collection[Any] = request.app.state.loan_applications
    llm_url: str = request.app.state.llm_url

    cust_oid = parse_object_id(customer_id, "customer_id")
    loan_oid = parse_object_id(loan_id, "loan_id")

    pipeline: List[dict[str, Any]] = [
        {"$match": {"_id": cust_oid}},
        {
            "$lookup": {
                "from": loans.name,
                "let": {"cid": "$_id"},
                "pipeline": [
                    {"$match": {"$expr": {"$and": [{"$eq": ["$customer_id", "$$cid"]}, {"$eq": ["$_id", loan_oid]}]}}},
                    {"$limit": 1},
                ],
                "as": "loan_docs",
            }
        },
        {"$addFields": {"loan": {"$first": "$loan_docs"}}},
        {"$project": {"loan_docs": 0}},
    ]

    rows = list(customers.aggregate(pipeline))
    if not rows:
        raise HTTPException(status_code=404, detail="Customer not found")
    row = rows[0]
    loan_doc = row.get("loan")
    if not loan_doc:
        raise HTTPException(status_code=404, detail="Loan application not found for this customer")

    customer_doc = {k: v for k, v in row.items() if k != "loan"}
    risk_score, risk_level, decision, factors = compute_risk_score(customer_doc, loan_doc)

    docs_coll: Collection[Any] = request.app.state.documents
    supporting_docs = list(docs_coll.find(
        {"customer_id": cust_oid},
        {"pdf_data": 0},
    ))

    doc_context_parts: list[str] = []
    for sd in supporting_docs:
        text = sd.get("text_content", "")
        if text:
            doc_context_parts.append(
                f"[{sd.get('doc_type', 'document').upper()}] {sd.get('filename', '')}\n{text}"
            )
    doc_context = "\n\n".join(doc_context_parts) if doc_context_parts else ""

    expl_user = _credit_explanation_prompt(
        customer_doc, loan_doc, risk_score, risk_level, decision, factors
    )
    if doc_context:
        expl_user += (
            "\n\nSupporting documents on file for this customer:\n"
            + doc_context
            + "\n\nReference specific document evidence (income figures, balances) in your explanation."
        )

    explanation = call_llm(
        [
            {"role": "system", "content": "You write clear, professional credit decision explanations."},
            {"role": "user", "content": expl_user},
        ],
        llm_url,
    )

    return {
        "customer": _serialize(customer_doc),
        "loan": _serialize(loan_doc),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "decision": decision,
        "explanation": explanation,
        "factors": factors,
        "supporting_documents": [
            {
                "_id": str(sd["_id"]),
                "doc_type": sd.get("doc_type"),
                "filename": sd.get("filename"),
                "generated_date": sd.get("generated_date"),
            }
            for sd in supporting_docs
        ],
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
    voyage_client: voyageai.Client = request.app.state.voyage
    kyc_coll: Collection[Any] = request.app.state.kyc_documents

    doc_oid = parse_object_id(document_id, "document_id")
    doc = kyc_coll.find_one({"_id": doc_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="KYC document not found")

    description = str(doc.get("description") or doc.get("summary") or "")
    if not description.strip():
        raise HTTPException(status_code=400, detail="Document has no embeddable description")

    query_vector = get_query_embedding(voyage_client, description)

    # Exclude current doc with $match after $vectorSearch — Atlas does not allow
    # filter on _id unless _id is declared as a filter field on the search index.
    pipeline: List[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": KYC_VECTOR_INDEX,
                "path": KYC_EMBEDDING_FIELD,
                "queryVector": query_vector,
                "numCandidates": 150,
                "limit": 20,
            }
        },
        {"$addFields": {"similarity": {"$meta": "vectorSearchScore"}}},
        {"$match": {"_id": {"$ne": doc_oid}}},
        {"$project": {KYC_EMBEDDING_FIELD: 0}},
    ]

    try:
        similar = list(kyc_coll.aggregate(pipeline))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Atlas Vector Search failed ({exc}). "
                f'Ensure Atlas Search index "{KYC_VECTOR_INDEX}" exists on '
                f'banking.kyc_documents with vector field "{KYC_EMBEDDING_FIELD}" '
                f"(see README / setup_workshop.py / 07_load_kyc_data.py)."
            ),
        ) from exc
    duplicates: List[dict[str, Any]] = []
    for s in similar:
        sim = s.get("similarity")
        sim_f = float(sim) if isinstance(sim, (int, float)) else 0.0
        if sim_f >= threshold:
            duplicates.append(
                {
                    "id": str(s.get("_id")),
                    "customer_id": str(s.get("customer_id")) if s.get("customer_id") else None,
                    "document_type": s.get("document_type"),
                    "similarity": round(sim_f, 6),
                }
            )

    expired = False
    risk_flags: List[str] = []
    expiry = doc.get("expiry_date") or doc.get("expires_at")
    today = date.today()
    if expiry:
        exp_date: Optional[date] = None
        if isinstance(expiry, datetime):
            exp_date = expiry.date()
        elif isinstance(expiry, date):
            exp_date = expiry
        elif isinstance(expiry, str):
            try:
                exp_date = datetime.fromisoformat(expiry.replace("Z", "+00:00")).date()
            except ValueError:
                risk_flags.append("invalid_expiry_format")
        if exp_date and exp_date < today:
            expired = True
            risk_flags.append("document_expired")

    if duplicates:
        risk_flags.append("possible_duplicate_documents")

    doc_out = dict(doc)
    doc_out.pop(KYC_EMBEDDING_FIELD, None)
    doc_out.pop("embedding_model", None)
    doc_out.pop("pdf_data", None)

    return {
        "document": _serialize(doc_out),
        "duplicates": duplicates,
        "expired": expired,
        "risk_flags": risk_flags,
    }


@app.get("/api/kyc-documents")
async def api_kyc_documents(request: Request) -> list[dict[str, Any]]:
    coll: Collection[Any] = request.app.state.kyc_documents
    projection = {KYC_EMBEDDING_FIELD: 0, "embedding_model": 0, "pdf_data": 0}
    return _serialize_cursor(coll.find({}, projection))


@app.get("/api/kyc-documents/{doc_id}/pdf")
async def api_kyc_document_pdf(request: Request, doc_id: str) -> Response:
    """Serve the scanned KYC document PDF."""
    coll: Collection[Any] = request.app.state.kyc_documents
    oid = parse_object_id(doc_id, "doc_id")
    doc = coll.find_one({"_id": oid})
    if not doc or "pdf_data" not in doc:
        raise HTTPException(status_code=404, detail="KYC document PDF not found")
    pdf_bytes = bytes(doc["pdf_data"])
    doc_type = doc.get("document_type", "document")
    doc_num = doc.get("document_number", doc_id)
    filename = f"{doc_type}_{doc_num}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


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
    voyage_client: voyageai.Client = request.app.state.voyage
    customers: Collection[Any] = request.app.state.customers
    products: Collection[Any] = request.app.state.bank_products

    cust_oid = parse_object_id(customer_id, "customer_id")
    customer = customers.find_one({"_id": cust_oid})
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    profile = _build_customer_profile_text(customer)
    query_vector = get_query_embedding(voyage_client, profile)

    pipeline: List[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": PRODUCT_VECTOR_INDEX,
                "path": EMBEDDING_FIELD,
                "queryVector": query_vector,
                "numCandidates": min(200, max(50, limit * 20)),
                "limit": limit,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {EMBEDDING_FIELD: 0}},
    ]

    recs: List[dict[str, Any]] = []
    for p in products.aggregate(pipeline):
        sc = p.get("score")
        recs.append(
            {
                "name": p.get("name", ""),
                "description": p.get("description", ""),
                "score": round(float(sc), 6) if isinstance(sc, (int, float)) else sc,
            }
        )

    return {"customer_id": customer_id, "recommendations": recs}


# ──────────────────────────────────────────────────────────
# Customer documents (payslips, bank statements)
# ──────────────────────────────────────────────────────────

@app.get("/api/documents")
async def api_documents(
    request: Request,
    customer_id: str = Query(...),
) -> list[dict[str, Any]]:
    """List documents for a customer (metadata only, no binary PDF)."""
    coll: Collection[Any] = request.app.state.documents
    cust_oid = parse_object_id(customer_id, "customer_id")
    docs = coll.find({"customer_id": cust_oid}, {"pdf_data": 0})
    return _serialize_cursor(docs)


@app.get("/api/documents/{doc_id}/pdf")
async def api_document_pdf(request: Request, doc_id: str) -> Response:
    """Download a single PDF document by its _id."""
    coll: Collection[Any] = request.app.state.documents
    oid = parse_object_id(doc_id, "doc_id")
    doc = coll.find_one({"_id": oid})
    if not doc or "pdf_data" not in doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pdf_bytes = bytes(doc["pdf_data"])
    filename = doc.get("filename", f"{doc_id}.pdf")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


# ──────────────────────────────────────────────────────────
# Debug: CloudWatch Lambda Logs + Bedrock evidence
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
# Admin endpoints — used by the Architecture tab to show live
# counts + sample documents proving the document model.
# ──────────────────────────────────────────────────────────

@app.get("/api/admin/collections")
async def api_admin_collections(request: Request) -> dict[str, Any]:
    """Live count of documents in every collection used by the demo."""
    db = request.app.state.db
    wanted = [
        "customers", "accounts", "transactions",
        "loan_applications", "loan_support_docs", "insurance_policies",
        "kyc_documents", "documents", "faq_chunks",
        "bank_products", "network_graph",
    ]
    counts: dict[str, int] = {}
    for name in wanted:
        try:
            counts[name] = db[name].estimated_document_count()
        except Exception:
            counts[name] = -1
    return {"database": "banking", "collections": counts}


@app.get("/api/admin/sample-document")
async def api_admin_sample_document(
    request: Request,
    collection: str = Query(..., min_length=1),
    redact_embedding: bool = Query(True, description="Trim embedding arrays to first 8 dims for display"),
) -> dict[str, Any]:
    """
    Return ONE real document from the requested collection — used by the
    Architecture page to show, transparently, that operational fields,
    relationships, and (where applicable) vector embeddings all live in
    the SAME document.
    """
    db = request.app.state.db
    # Whitelist to avoid leaking unintended internal collections
    allowed = {
        "customers", "accounts", "transactions",
        "loan_applications", "loan_support_docs", "insurance_policies",
        "kyc_documents", "documents", "faq_chunks",
        "bank_products", "network_graph",
    }
    if collection not in allowed:
        raise HTTPException(status_code=400, detail=f"collection not allowed: {collection}")

    doc = db[collection].find_one() or {}
    # Pick a document with an embedding when possible — this is the highlight.
    if "embedding" not in doc and "description_embedding" not in doc:
        for cand in db[collection].find(
            {"$or": [{"embedding": {"$exists": True}}, {"description_embedding": {"$exists": True}}]}
        ).limit(1):
            doc = cand
            break

    # Trim embeddings to keep payload sane and visually readable.
    trimmed = _serialize(doc)
    embedding_meta: dict[str, Any] = {}
    for emb_key in ("embedding", "description_embedding"):
        if emb_key in trimmed and isinstance(trimmed[emb_key], list):
            full_len = len(trimmed[emb_key])
            embedding_meta[emb_key] = {
                "dims": full_len,
                "preview_first_8": trimmed[emb_key][:8],
                "model_field": trimmed.get("embedding_model"),
            }
            if redact_embedding:
                trimmed[emb_key] = trimmed[emb_key][:8] + [f"... +{full_len - 8} more dims"]
    # Also redact heavy binary blobs (PDFs stored inline) so the JSON is readable.
    for blob_key in ("pdf_data", "binary", "image"):
        if blob_key in trimmed and isinstance(trimmed[blob_key], (bytes, str)) and len(str(trimmed[blob_key])) > 200:
            trimmed[blob_key] = f"<binary blob, {len(str(trimmed[blob_key]))} chars — redacted for display>"

    # Quick anatomy of the document — useful for the "single-doc advantage" panel
    anatomy = {
        "top_level_field_count": len(trimmed.keys()),
        "has_embedding": any(k in doc for k in ("embedding", "description_embedding")),
        "embedded_array_lengths": {
            k: len(v) for k, v in doc.items() if isinstance(v, list)
        },
        "nested_object_keys": {
            k: list(v.keys())[:8] for k, v in doc.items() if isinstance(v, dict) and not isinstance(v, list)
        },
    }

    return {
        "collection": f"banking.{collection}",
        "document": trimmed,
        "embedding_meta": embedding_meta,
        "anatomy": anatomy,
    }


@app.get("/api/debug/lambda-logs")
async def api_lambda_logs(
    minutes: int = Query(10, ge=1, le=60),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Fetch recent BEDROCK_REQUEST / BEDROCK_RESPONSE logs from CloudWatch."""
    try:
        import boto3
    except ImportError:
        raise HTTPException(status_code=501, detail="boto3 not installed")

    log_group = "/aws/lambda/fsi_workshop"
    start_ms = int((time.time() - minutes * 60) * 1000)

    try:
        cw = boto3.client("logs", region_name="us-east-1")
        resp = cw.filter_log_events(
            logGroupName=log_group,
            startTime=start_ms,
            limit=limit,
            interleaved=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CloudWatch error: {exc}")

    raw_events = []
    bedrock_events = []
    for ev in resp.get("events", []):
        msg = ev.get("message", "").strip()
        if not msg:
            continue
        ts = ev.get("timestamp")
        if msg.startswith("REPORT"):
            parts = msg.split("\t")
            raw_events.append({"timestamp": ts, "type": "REPORT", "message": msg})
            continue
        if msg.startswith(("START", "END", "INIT_START")):
            continue
        try:
            parsed = json.loads(msg.split("\t")[-1] if "\t" in msg else msg)
            if isinstance(parsed, dict) and parsed.get("event", "").startswith("BEDROCK_"):
                parsed["_timestamp"] = ts
                bedrock_events.append(parsed)
                continue
        except (json.JSONDecodeError, IndexError):
            pass
        raw_events.append({"timestamp": ts, "type": "LOG", "message": msg[:500]})

    return {
        "log_group": log_group,
        "last_minutes": minutes,
        "bedrock_events": bedrock_events,
        "raw_events": raw_events,
    }


# ──────────────────────────────────────────────────────────
# Customer Intelligence: 360 view + AI marketing insights
# ──────────────────────────────────────────────────────────

CUSTOMER_INTEL_PROMPT = (
    "You are a senior banking marketing strategist and data analyst.\n"
    "Analyze this customer's complete profile and generate actionable marketing intelligence.\n\n"
    "Return ONLY valid JSON (no markdown fences) with this structure:\n"
    "{\n"
    '  "segment": "<segment name e.g. High-Value Professional, Young Saver, etc.>",\n'
    '  "segment_description": "<1 sentence explaining why this segment>",\n'
    '  "lifetime_value_tier": "<platinum|gold|silver|bronze>",\n'
    '  "campaigns": [\n'
    '    {"name": "<campaign name>", "channel": "<email|sms|in_app|branch>",\n'
    '     "message": "<the actual personalized marketing message (2-3 sentences)>",\n'
    '     "rationale": "<why this campaign for this customer>",\n'
    '     "expected_conversion": "<high|medium|low>"}\n'
    "  ],\n"
    '  "cross_sell": ["<product/service opportunity>", ...],\n'
    '  "next_best_action": "<single most impactful action to take with this customer>",\n'
    '  "risk_of_churn": "<high|medium|low>",\n'
    '  "churn_factors": ["<factor>", ...],\n'
    '  "key_insights": ["<insight about this customer>", ...]\n'
    "}\n\n"
    "Generate 3-4 campaigns. Be specific with dollar amounts, product names, and personalization.\n"
    "Reference the customer's actual data (income, credit score, transaction patterns, goals) in your reasoning."
)


@app.get("/api/customer-intelligence")
async def api_customer_intelligence(
    request: Request,
    customer_id: str = Query(...),
) -> dict[str, Any]:
    """
    Build a full Customer 360 view using MongoDB $lookup aggregation,
    find matching products via $vectorSearch, and generate AI-powered
    marketing intelligence using Claude.
    """
    db = request.app.state.db
    voyage_client: voyageai.Client = request.app.state.voyage
    llm_url: str = request.app.state.llm_url

    cust_oid = parse_object_id(customer_id, "customer_id")

    # ── Step 1: Customer 360 aggregation ($lookup) ──
    t_agg = time.time()
    pipeline_360: List[dict[str, Any]] = [
        {"$match": {"_id": cust_oid}},
        {
            "$lookup": {
                "from": "accounts",
                "localField": "_id",
                "foreignField": "customer_id",
                "as": "accounts",
            }
        },
        {
            "$lookup": {
                "from": "loan_applications",
                "localField": "_id",
                "foreignField": "customer_id",
                "as": "loans",
            }
        },
        {
            "$lookup": {
                "from": "transactions",
                "localField": "_id",
                "foreignField": "customer_id",
                "pipeline": [{"$sort": {"date": -1}}, {"$limit": 30}],
                "as": "recent_transactions",
            }
        },
        {
            "$lookup": {
                "from": "kyc_documents",
                "let": {"cid": {"$toString": "$_id"}},
                "pipeline": [
                    {"$match": {"$expr": {"$eq": ["$customer_id", "$$cid"]}}},
                ],
                "as": "kyc_docs",
            }
        },
    ]
    rows = list(db.customers.aggregate(pipeline_360))
    agg_ms = round((time.time() - t_agg) * 1000)

    if not rows:
        raise HTTPException(status_code=404, detail="Customer not found")
    customer_360 = rows[0]

    # Compute derived metrics
    accounts = customer_360.get("accounts", [])
    loans = customer_360.get("loans", [])
    txns = customer_360.get("recent_transactions", [])
    total_balance = sum(float(a.get("balance", 0)) for a in accounts)
    total_loan_amount = sum(float(l.get("loan_amount", 0)) for l in loans)
    avg_txn = sum(float(t.get("amount", 0)) for t in txns) / max(len(txns), 1)
    txn_categories = {}
    for t in txns:
        cat = t.get("category", "other")
        txn_categories[cat] = txn_categories.get(cat, 0) + 1

    profile_summary = {
        "name": customer_360.get("full_name", ""),
        "city": customer_360.get("city", ""),
        "country": customer_360.get("country", ""),
        "employment": customer_360.get("employment_type", ""),
        "monthly_income": customer_360.get("monthly_income", 0),
        "credit_score": customer_360.get("credit_score", 0),
        "payment_history": customer_360.get("payment_history", ""),
        "financial_goals": customer_360.get("financial_goals", ""),
        "account_age_months": customer_360.get("account_age_months", 0),
        "num_accounts": len(accounts),
        "total_balance": round(total_balance, 2),
        "num_loans": len(loans),
        "total_loan_amount": round(total_loan_amount, 2),
        "loan_statuses": [l.get("status") for l in loans],
        "loan_purposes": [l.get("purpose") for l in loans],
        "recent_txn_count": len(txns),
        "avg_txn_amount": round(avg_txn, 2),
        "top_txn_categories": dict(sorted(txn_categories.items(), key=lambda x: -x[1])[:5]),
        "kyc_status": [d.get("verification_status") for d in customer_360.get("kyc_docs", [])],
    }

    # ── Step 2: Vector search for matching products ──
    profile_text = _build_customer_profile_text(customer_360)
    t_vs = time.time()
    query_vector = get_query_embedding(voyage_client, profile_text)
    embed_ms = round((time.time() - t_vs) * 1000)

    vs_pipeline: List[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": PRODUCT_VECTOR_INDEX,
                "path": EMBEDDING_FIELD,
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 5,
            }
        },
        {"$addFields": {"match_score": {"$meta": "vectorSearchScore"}}},
        {"$project": {EMBEDDING_FIELD: 0}},
    ]
    t_vs2 = time.time()
    matched_products = list(db.bank_products.aggregate(vs_pipeline))
    vs_ms = round((time.time() - t_vs2) * 1000)

    product_summaries = [
        {"name": p.get("name", ""), "description": p.get("description", "")[:200], "score": round(float(p.get("match_score", 0)), 4)}
        for p in matched_products
    ]

    # ── Step 3: Claude generates marketing intelligence ──
    user_content = (
        f"Customer profile:\n{json.dumps(profile_summary, indent=2, default=str)}\n\n"
        f"Vector-search matched products (ranked by relevance):\n{json.dumps(product_summaries, indent=2)}\n\n"
        "Generate comprehensive marketing intelligence for this customer."
    )
    messages = [
        {"role": "system", "content": CUSTOMER_INTEL_PROMPT},
        {"role": "user", "content": user_content},
    ]
    t_llm = time.time()
    try:
        resp = requests.post(llm_url, json={"messages": messages}, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
    llm_ms = round((time.time() - t_llm) * 1000)

    llm_data = resp.json()
    raw_answer = (llm_data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    usage = llm_data.get("usage", {})

    intel: dict[str, Any] = {}
    try:
        intel = json.loads(raw_answer)
    except json.JSONDecodeError:
        cleaned = raw_answer
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].rsplit("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].rsplit("```", 1)[0]
        try:
            intel = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            intel = {"error": "Failed to parse AI response", "raw": raw_answer[:2000]}

    return _serialize({
        "customer": profile_summary,
        "matched_products": product_summaries,
        "intelligence": intel,
        "pipeline": {
            "aggregation_ms": agg_ms,
            "embed_ms": embed_ms,
            "vector_search_ms": vs_ms,
            "llm_ms": llm_ms,
            "usage": usage,
            "stages": [
                "$match → $lookup(accounts) → $lookup(loans) → $lookup(transactions) → $lookup(kyc)",
                f"$vectorSearch(product_vector_index, {len(query_vector)}-dim query)",
                "Claude: profile + products → marketing intelligence",
            ],
        },
    })


# ──────────────────────────────────────────────────────────
# Analytics Dashboard: MongoDB aggregation pipelines
# ──────────────────────────────────────────────────────────

@app.get("/api/analytics")
async def api_analytics(request: Request) -> dict[str, Any]:
    """Run multiple aggregation pipelines and return dashboard metrics."""
    db = request.app.state.db

    # 1. Loan status breakdown
    loan_status = list(db.loan_applications.aggregate([
        {"$group": {"_id": "$status", "count": {"$sum": 1}, "total_amount": {"$sum": "$loan_amount"}}},
        {"$sort": {"count": -1}},
    ]))

    # 2. Loan purpose distribution
    loan_by_purpose = list(db.loan_applications.aggregate([
        {"$group": {"_id": "$purpose", "count": {"$sum": 1}, "avg_amount": {"$avg": "$loan_amount"}}},
        {"$sort": {"count": -1}},
        {"$limit": 10},
    ]))

    # 3. Risk level distribution via customer credit scores
    risk_distribution = list(db.customers.aggregate([
        {"$bucket": {
            "groupBy": "$credit_score",
            "boundaries": [0, 400, 600, 700, 850],
            "default": "unknown",
            "output": {"count": {"$sum": 1}, "avg_income": {"$avg": "$monthly_income"}},
        }},
    ]))

    # 4. Top-level KPIs
    total_customers = db.customers.count_documents({})
    total_loans = db.loan_applications.count_documents({})
    total_kyc = db.kyc_documents.count_documents({})

    agg_loan_totals = list(db.loan_applications.aggregate([
        {"$group": {"_id": None, "total": {"$sum": "$loan_amount"}, "avg": {"$avg": "$loan_amount"}}},
    ]))
    loan_totals = agg_loan_totals[0] if agg_loan_totals else {}

    # 5. Employment type breakdown
    employment = list(db.customers.aggregate([
        {"$group": {"_id": "$employment_type", "count": {"$sum": 1}, "avg_credit_score": {"$avg": "$credit_score"}}},
        {"$sort": {"count": -1}},
    ]))

    # 6. Monthly loan trend (by application_date)
    loan_trend = list(db.loan_applications.aggregate([
        {"$addFields": {"month": {"$dateToString": {"format": "%Y-%m", "date": "$application_date"}}}},
        {"$group": {"_id": "$month", "count": {"$sum": 1}, "volume": {"$sum": "$loan_amount"}}},
        {"$sort": {"_id": 1}},
        {"$limit": 12},
    ]))

    return _serialize({
        "kpis": {
            "total_customers": total_customers,
            "total_loans": total_loans,
            "total_kyc_documents": total_kyc,
            "total_loan_amount": loan_totals.get("total", 0),
            "avg_loan_amount": loan_totals.get("avg", 0),
        },
        "loan_status": loan_status,
        "loan_by_purpose": loan_by_purpose,
        "risk_distribution": risk_distribution,
        "employment": employment,
        "loan_trend": loan_trend,
    })


# ──────────────────────────────────────────────────────────
# Graph / Network Analysis: $graphLookup over the network_graph
# collection. Powers fraud-ring detection, AML transaction tracing,
# insurance beneficiary networks and loan-guarantor concentration.
# ──────────────────────────────────────────────────────────

GRAPH_COL_NAME = "network_graph"

# Maximum nodes returned to the frontend in any single response.
# vis-network handles ~500 well; beyond that the layout becomes unreadable.
GRAPH_NODE_LIMIT = 600


def _serialize_node(doc: dict[str, Any], hops: int | None = None,
                    is_root: bool = False) -> dict[str, Any]:
    """Shape a network_graph document for the frontend visualizer."""
    return {
        "id": doc["_id"],
        "label": doc.get("label") or doc["_id"],
        "type": doc.get("type", "Unknown"),
        "metadata": doc.get("metadata", {}),
        "doc_count": len(doc.get("relationships", [])),
        "hops": hops,
        "is_root": is_root,
    }


def _flatten_edges(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a node's embedded relationships[] into edge objects."""
    out = []
    for rel in doc.get("relationships", []) or []:
        edge = {"from": doc["_id"], "to": rel.get("target"), "label": rel.get("type", "")}
        # surface meaningful edge metadata for tooltips, INCLUDING `source`
        # (the human-readable provenance of where this edge came from)
        for k in ("source", "amount_usd", "transaction_count", "total_amount_usd",
                  "date", "chain_step", "loan_id", "description"):
            if k in rel:
                edge[k] = rel[k]
        out.append(edge)
    return out


@app.get("/api/graph/overview")
async def api_graph_overview(
    request: Request,
    sample_size: int = Query(200, ge=10, le=GRAPH_NODE_LIMIT),
) -> dict[str, Any]:
    """
    Sampled subgraph for first paint, augmented with auto-detected risk alerts
    and the full set of suspicious node IDs so the frontend can highlight them.
    Always includes every flagged node (watchlist, AML-flagged, shared
    address/phone/device) so risk patterns are visible immediately.
    """
    graph: Collection[Any] = request.app.state.network_graph
    t0 = time.time()

    # ── Step 1: detect alerts to know which nodes MUST appear ──
    alerts = _detect_risk_alerts(graph)
    must_have_ids: set[str] = set()
    for a in alerts:
        must_have_ids.update(a.get("highlight_nodes") or [])
        if a.get("focus_node"):
            must_have_ids.add(a["focus_node"])

    # ── Step 2: fetch must-have nodes ──
    must_have_docs = list(graph.find({"_id": {"$in": list(must_have_ids)}})) if must_have_ids else []

    # ── Step 3: sample remaining nodes (excluding must-haves) ──
    remaining_size = max(0, sample_size - len(must_have_docs))
    sample_docs = []
    if remaining_size > 0:
        sample_pipeline = [
            {"$match": {"_id": {"$nin": list(must_have_ids)}}},
            {"$addFields": {"_conn": {"$size": {"$ifNull": ["$relationships", []]}}}},
            {"$match": {"_conn": {"$gte": 1}}},
            {"$sample": {"size": remaining_size}},
        ]
        sample_docs = list(graph.aggregate(sample_pipeline))

    all_docs = must_have_docs + sample_docs
    if not all_docs:
        return {"nodes": [], "edges": [], "alerts": [], "query_time_ms": 0,
                "error": "network_graph is empty — run python setup_graph.py first"}

    # ── Step 4: serialize & flag each node ──
    flagged_set = must_have_ids
    sample_ids = {d["_id"] for d in all_docs}
    nodes = []
    for d in all_docs:
        n = _serialize_node(d)
        n["risk_flagged"] = d["_id"] in flagged_set
        nodes.append(n)

    edges = []
    for d in all_docs:
        for e in _flatten_edges(d):
            if e["to"] in sample_ids:
                edges.append(e)

    return {
        "nodes": nodes,
        "edges": edges,
        "alerts": alerts,
        "query_time_ms": round((time.time() - t0) * 1000, 1),
        "sample_size": sample_size,
        "total_nodes_in_db": graph.estimated_document_count(),
        "flagged_count": len(must_have_ids),
    }


@app.get("/api/graph/entity/{entity_id}")
async def api_graph_entity(
    request: Request,
    entity_id: str,
    depth: int = Query(2, ge=1, le=4),
) -> dict[str, Any]:
    """
    Multi-hop traversal from a single entity using $graphLookup.
    Returns the root entity + everything reachable within `depth` hops.
    """
    graph: Collection[Any] = request.app.state.network_graph
    root_doc = graph.find_one({"_id": entity_id})
    if not root_doc:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")

    t0 = time.time()
    pipeline = [
        {"$match": {"_id": entity_id}},
        {
            "$graphLookup": {
                "from": GRAPH_COL_NAME,
                "startWith": "$relationships.target",
                "connectFromField": "relationships.target",
                "connectToField": "_id",
                "as": "connected",
                "maxDepth": depth - 1,  # hops 0..depth-1 in the connected[] array
                "depthField": "hops",
            }
        },
    ]
    result = list(graph.aggregate(pipeline))
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    if not result:
        return {"nodes": [], "edges": [], "query_time_ms": elapsed_ms}

    root = result[0]
    connected = root.get("connected", [])

    # Build node + edge list
    nodes = [_serialize_node(root, hops=0, is_root=True)]
    seen_ids = {root["_id"]}
    for c in connected:
        if c["_id"] in seen_ids:
            continue
        nodes.append(_serialize_node(c, hops=int(c.get("hops", 0)) + 1))
        seen_ids.add(c["_id"])
        if len(nodes) >= GRAPH_NODE_LIMIT:
            break

    # Edges: include all edges where both endpoints are in our node set
    edges = []
    for d in [root] + connected:
        if d["_id"] not in seen_ids:
            continue
        for e in _flatten_edges(d):
            if e["to"] in seen_ids:
                edges.append(e)

    return {
        "root": entity_id,
        "root_label": root.get("label", entity_id),
        "root_type": root.get("type"),
        "nodes": nodes,
        "edges": edges,
        "total_connected": len(connected),
        "query_time_ms": elapsed_ms,
        "pipeline_used": {
            "description": "MongoDB $graphLookup — multi-hop traversal of relationships[]",
            "stages": [
                {"$match": {"_id": entity_id}},
                {
                    "$graphLookup": {
                        "from": GRAPH_COL_NAME,
                        "startWith": "$relationships.target",
                        "connectFromField": "relationships.target",
                        "connectToField": "_id",
                        "as": "connected",
                        "maxDepth": depth - 1,
                        "depthField": "hops",
                    }
                },
            ],
        },
    }


@app.get("/api/graph/search")
async def api_graph_search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=80),
    limit: int = Query(20, ge=1, le=50),
) -> dict[str, Any]:
    """Autocomplete-style search by node label or _id."""
    graph: Collection[Any] = request.app.state.network_graph
    rx = {"$regex": re.escape(q), "$options": "i"}
    cursor = graph.find(
        {"$or": [{"_id": rx}, {"label": rx}]},
        {"_id": 1, "label": 1, "type": 1, "metadata": 1}
    ).limit(limit)
    return {"results": [
        {
            "id": d["_id"],
            "label": d.get("label", d["_id"]),
            "type": d.get("type"),
            "metadata": d.get("metadata", {}),
        }
        for d in cursor
    ]}


@app.post("/api/graph/scenario")
async def api_graph_scenario(
    request: Request,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Pre-built investigative scenarios. Each one runs a focused $graphLookup
    starting from a meaningful entity and returns a graph + narrative.
    """
    graph: Collection[Any] = request.app.state.network_graph
    scenario = (payload or {}).get("scenario", "")
    t0 = time.time()

    # ─── Scenario 1: Fraud Ring Detection ──────────────────
    if scenario == "fraud_ring":
        # Find Address/Phone/Device nodes shared by multiple customers,
        # pick the one with the most customers, traverse outward.
        candidate = graph.find_one(
            {"type": {"$in": ["Address", "Phone", "Device"]},
             "metadata.shared_by": {"$gte": 2}},
            sort=[("metadata.shared_by", -1)],
        )
        if not candidate:
            return {"error": "No shared address/phone/device found", "nodes": [], "edges": []}

        pipeline = [
            {"$match": {"_id": candidate["_id"]}},
            {
                "$graphLookup": {
                    "from": GRAPH_COL_NAME,
                    "startWith": "$relationships.target",
                    "connectFromField": "relationships.target",
                    "connectToField": "_id",
                    "as": "connected",
                    "maxDepth": 2,
                    "depthField": "hops",
                }
            },
        ]
        result = list(graph.aggregate(pipeline))
        return _build_scenario_response(
            result, candidate["_id"], pipeline, t0,
            title="Fraud Ring Detection",
            narrative=(
                f"Started from shared {candidate.get('type','').lower()} "
                f"'{candidate.get('label')}' which is registered to "
                f"{candidate.get('metadata',{}).get('shared_by','?')} customers. "
                "Traversing 2 hops surfaces all of their accounts, loans and policies "
                "— a synthetic-identity ring fingerprint."
            ),
        )

    # ─── Scenario 2: AML Transaction Trace ─────────────────
    if scenario == "aml_trace":
        # Find the first WIRE_SENT edge in the AML chain (chain_step=1).
        # Use $elemMatch so chain_step AND type must match in the SAME relationship.
        seed = graph.find_one(
            {"relationships": {"$elemMatch": {"chain_step": 1, "type": "WIRE_SENT"}}},
        )
        if not seed:
            return {"error": "No AML chain seed found", "nodes": [], "edges": []}

        pipeline = [
            {"$match": {"_id": seed["_id"]}},
            {
                "$graphLookup": {
                    "from": GRAPH_COL_NAME,
                    "startWith": "$relationships.target",
                    "connectFromField": "relationships.target",
                    "connectToField": "_id",
                    "as": "connected",
                    "maxDepth": 6,
                    "depthField": "hops",
                    "restrictSearchWithMatch": {
                        "$or": [
                            {"type": {"$in": ["Account", "Customer"]}},
                        ]
                    },
                }
            },
        ]
        result = list(graph.aggregate(pipeline))
        return _build_scenario_response(
            result, seed["_id"], pipeline, t0,
            title="AML Transaction Trace",
            narrative=(
                f"Tracing structured wire transfers from account {seed['_id']}. "
                "Each hop is just under the $10k reporting threshold — classic "
                "layering pattern. $graphLookup walks the chain and surfaces every "
                "account and customer touched by the funds."
            ),
        )

    # ─── Scenario 3: Insurance Beneficiary Network ────────
    if scenario == "beneficiary_network":
        # Find the customer named as beneficiary on the most policies
        agg = list(graph.aggregate([
            {"$match": {"type": "InsurancePolicy"}},
            {"$unwind": "$relationships"},
            {"$match": {"relationships.type": "BENEFICIARY_IS"}},
            {"$group": {"_id": "$relationships.target", "policy_count": {"$sum": 1}}},
            {"$sort": {"policy_count": -1}},
            {"$limit": 1},
        ]))
        if not agg:
            return {"error": "No beneficiaries found", "nodes": [], "edges": []}
        focal = agg[0]["_id"]
        focal_count = agg[0]["policy_count"]

        pipeline = [
            {"$match": {"_id": focal}},
            {
                "$graphLookup": {
                    "from": GRAPH_COL_NAME,
                    "startWith": "$relationships.target",
                    "connectFromField": "relationships.target",
                    "connectToField": "_id",
                    "as": "connected",
                    "maxDepth": 2,
                    "depthField": "hops",
                }
            },
        ]
        result = list(graph.aggregate(pipeline))
        return _build_scenario_response(
            result, focal, pipeline, t0,
            title="Insurance Beneficiary Network",
            narrative=(
                f"Customer {focal} is named as beneficiary on {focal_count} policies "
                "across multiple unrelated holders — a 'ghost beneficiary' pattern "
                "warranting review. The graph shows every policy holder pointing into "
                "this single beneficiary."
            ),
        )

    # ─── Scenario 4: Loan Guarantor Concentration ─────────
    if scenario == "loan_guarantor":
        # Find the customer guaranteeing the most loans
        agg = list(graph.aggregate([
            {"$match": {"type": "Customer"}},
            {"$addFields": {
                "guarantor_count": {
                    "$size": {
                        "$filter": {
                            "input": "$relationships",
                            "cond": {"$eq": ["$$this.type", "GUARANTOR_OF"]}
                        }
                    }
                }
            }},
            {"$match": {"guarantor_count": {"$gte": 2}}},
            {"$sort": {"guarantor_count": -1}},
            {"$limit": 1},
        ]))
        if not agg:
            return {"error": "No multi-loan guarantor found", "nodes": [], "edges": []}
        focal = agg[0]["_id"]
        gcount = agg[0]["guarantor_count"]

        pipeline = [
            {"$match": {"_id": focal}},
            {
                "$graphLookup": {
                    "from": GRAPH_COL_NAME,
                    "startWith": "$relationships.target",
                    "connectFromField": "relationships.target",
                    "connectToField": "_id",
                    "as": "connected",
                    "maxDepth": 2,
                    "depthField": "hops",
                }
            },
        ]
        result = list(graph.aggregate(pipeline))
        return _build_scenario_response(
            result, focal, pipeline, t0,
            title="Loan Guarantor Concentration",
            narrative=(
                f"{focal} is the guarantor on {gcount} active loans. If this person "
                "defaults or loses income, all guaranteed loans become at-risk simultaneously. "
                "$graphLookup walks the guarantor edge to expose the borrower network."
            ),
        )

    raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario}")


def _build_scenario_response(
    aggregate_result: list[dict],
    root_id: str,
    pipeline: list[dict],
    t0: float,
    title: str,
    narrative: str,
) -> dict[str, Any]:
    """Shared response shaper for all scenarios."""
    elapsed_ms = round((time.time() - t0) * 1000, 1)
    if not aggregate_result:
        return {"title": title, "narrative": narrative, "nodes": [], "edges": [],
                "query_time_ms": elapsed_ms}

    root = aggregate_result[0]
    connected = root.get("connected", [])

    nodes = [_serialize_node(root, hops=0, is_root=True)]
    seen = {root["_id"]}
    for c in connected:
        if c["_id"] in seen:
            continue
        nodes.append(_serialize_node(c, hops=int(c.get("hops", 0)) + 1))
        seen.add(c["_id"])
        if len(nodes) >= GRAPH_NODE_LIMIT:
            break

    edges = []
    for d in [root] + connected:
        if d["_id"] not in seen:
            continue
        for e in _flatten_edges(d):
            if e["to"] in seen:
                edges.append(e)

    # Convert pipeline to JSON-serializable form for display
    return {
        "title": title,
        "narrative": narrative,
        "root": root_id,
        "root_label": root.get("label", root_id),
        "root_type": root.get("type"),
        "nodes": nodes,
        "edges": edges,
        "total_connected": len(connected),
        "query_time_ms": elapsed_ms,
        "pipeline_used": {
            "description": f"$graphLookup pipeline for {title}",
            "stages": pipeline,
        },
    }


# ──────────────────────────────────────────────────────────
# Risk Alerts: deterministic detection of suspicious patterns.
# Runs once per overview load; the frontend uses this to highlight
# nodes and surface a clickable alert banner above the graph.
# ──────────────────────────────────────────────────────────

def _detect_risk_alerts(graph: Collection[Any]) -> list[dict[str, Any]]:
    """
    Run aggregation queries against network_graph to surface every detectable
    pattern. Each alert includes the node IDs to highlight in the visualization.
    """
    alerts: list[dict[str, Any]] = []

    # 1) Synthetic identity / mule clusters: shared address, phone or device
    shared = list(graph.aggregate([
        {"$match": {
            "type": {"$in": ["Address", "Phone", "Device"]},
            "metadata.shared_by": {"$gte": 2},
        }},
        {"$lookup": {
            "from": GRAPH_COL_NAME,
            "localField": "_id",
            "foreignField": "relationships.target",
            "as": "linked_back",
        }},
        {"$project": {
            "_id": 1, "type": 1, "label": 1,
            "shared_by": "$metadata.shared_by",
            "customers": {
                "$map": {
                    "input": {"$filter": {"input": "$linked_back",
                                          "cond": {"$eq": ["$$this.type", "Customer"]}}},
                    "as": "c", "in": "$$c._id",
                }
            },
        }},
        {"$sort": {"shared_by": -1}},
    ]))
    for s in shared:
        kind = s.get("type", "").lower()
        severity = "high" if s.get("shared_by", 0) >= 3 else "medium"
        node_ids = [s["_id"]] + (s.get("customers") or [])
        alerts.append({
            "id": f"alert_{s['_id']}",
            "kind": "synthetic_identity_ring" if kind in ("address", "device") else "shared_contact",
            "severity": severity,
            "title": f"Shared {kind} flagged on {s.get('shared_by')} customers",
            "summary": f"\"{s.get('label')}\" links {', '.join(s.get('customers', []))}",
            "highlight_nodes": node_ids,
            "focus_node": s["_id"],
        })

    # 2) AML chain: any node with relationships marked chain_step
    aml = list(graph.aggregate([
        {"$match": {"relationships": {"$elemMatch": {"chain_step": {"$exists": True}}}}},
        {"$project": {"_id": 1, "label": 1, "type": 1,
                      "wires": {"$filter": {"input": "$relationships",
                                            "cond": {"$ifNull": ["$$this.chain_step", False]}}}}},
        {"$unwind": "$wires"},
        {"$match": {"wires.type": "WIRE_SENT"}},
        {"$group": {
            "_id": None,
            "accounts": {"$addToSet": "$_id"},
            "targets": {"$addToSet": "$wires.target"},
            "hops": {"$sum": 1},
            "total_amount": {"$sum": "$wires.amount_usd"},
            "first_step": {"$min": {"$cond": [{"$eq": ["$wires.chain_step", 1]}, "$_id", None]}},
        }},
    ]))
    if aml:
        a = aml[0]
        all_accts = list(set((a.get("accounts") or []) + (a.get("targets") or [])))
        # Also include the account-owners (customers) for highlighting
        cust_links = list(graph.find({
            "type": "Customer",
            "relationships": {"$elemMatch": {"target": {"$in": all_accts}, "type": "OWNS_ACCOUNT"}}
        }, {"_id": 1}))
        cust_ids = [c["_id"] for c in cust_links]
        alerts.append({
            "id": "alert_aml_chain",
            "kind": "aml_layering",
            "severity": "high",
            "title": f"AML layering chain — {a['hops']} structured wires under threshold",
            "summary": (
                f"Total moved: ${a['total_amount']:,.0f} across {len(all_accts)} accounts. "
                f"Each transfer just under the $10k reporting threshold."
            ),
            "highlight_nodes": all_accts + cust_ids,
            "focus_node": a.get("first_step") or (all_accts[0] if all_accts else None),
        })

    # 3) Ghost beneficiary: customer named beneficiary on >= 3 unrelated policies
    ghost = list(graph.aggregate([
        {"$match": {"type": "InsurancePolicy"}},
        {"$unwind": "$relationships"},
        {"$match": {"relationships.type": "BENEFICIARY_IS"}},
        {"$group": {"_id": "$relationships.target",
                    "policy_count": {"$sum": 1},
                    "policy_ids": {"$push": "$_id"}}},
        {"$match": {"policy_count": {"$gte": 3}}},
        {"$lookup": {"from": GRAPH_COL_NAME, "localField": "_id",
                     "foreignField": "_id", "as": "cust"}},
        {"$unwind": "$cust"},
        {"$sort": {"policy_count": -1}},
    ]))
    for g in ghost:
        alerts.append({
            "id": f"alert_ghost_{g['_id']}",
            "kind": "ghost_beneficiary",
            "severity": "high" if g["policy_count"] >= 5 else "medium",
            "title": f"Ghost beneficiary — named on {g['policy_count']} policies",
            "summary": f"{g['cust'].get('label')} ({g['_id']}) is beneficiary on {g['policy_count']} unrelated policies.",
            "highlight_nodes": [g["_id"]] + g["policy_ids"],
            "focus_node": g["_id"],
        })

    # 4) Guarantor concentration: customer guaranteeing >= 3 loans
    guar = list(graph.aggregate([
        {"$match": {"type": "Customer"}},
        {"$addFields": {"guar_loans": {
            "$map": {
                "input": {"$filter": {"input": "$relationships",
                                      "cond": {"$eq": ["$$this.type", "GUARANTOR_OF"]}}},
                "as": "r", "in": "$$r.target",
            }
        }}},
        {"$match": {"guar_loans.0": {"$exists": True}}},
        {"$addFields": {"guar_count": {"$size": "$guar_loans"}}},
        {"$match": {"guar_count": {"$gte": 3}}},
        {"$sort": {"guar_count": -1}},
    ]))
    for g in guar:
        alerts.append({
            "id": f"alert_guar_{g['_id']}",
            "kind": "guarantor_concentration",
            "severity": "medium" if g["guar_count"] < 5 else "high",
            "title": f"Guarantor concentration — {g['guar_count']} active loans",
            "summary": f"{g.get('label')} ({g['_id']}) is the named guarantor on {g['guar_count']} loans.",
            "highlight_nodes": [g["_id"]] + g["guar_loans"],
            "focus_node": g["_id"],
        })

    return alerts


@app.get("/api/graph/alerts")
async def api_graph_alerts(request: Request) -> dict[str, Any]:
    """List every detectable risk pattern in the network_graph."""
    graph: Collection[Any] = request.app.state.network_graph
    t0 = time.time()
    alerts = _detect_risk_alerts(graph)
    return {
        "alerts": alerts,
        "total": len(alerts),
        "by_severity": {
            "high": sum(1 for a in alerts if a["severity"] == "high"),
            "medium": sum(1 for a in alerts if a["severity"] == "medium"),
        },
        "query_time_ms": round((time.time() - t0) * 1000, 1),
    }


# ──────────────────────────────────────────────────────────
# Conversational Investigation: LLM-powered chat over the graph.
# The LLM decides what to do with each user message and the backend
# executes a real $graphLookup or aggregation, then narrates results.
# ──────────────────────────────────────────────────────────

def _build_graph_catalog(graph: Collection[Any], max_nodes: int = 400) -> str:
    """
    Compact, prompt-friendly listing of every node so the LLM can resolve
    user references like 'the ghost beneficiary' or 'CUST 7' to a real _id.
    Format is one line per node: ID | type | label | key-meta
    """
    lines = []
    cur = graph.find(
        {},
        {"_id": 1, "type": 1, "label": 1, "metadata": 1},
    ).limit(max_nodes)
    for d in cur:
        m = d.get("metadata", {}) or {}
        meta_bits = []
        if m.get("watchlist"):       meta_bits.append("watchlist")
        if m.get("aml_flag"):        meta_bits.append("AML")
        if m.get("shared_by", 0) >= 2: meta_bits.append(f"shared_by_{m['shared_by']}")
        if m.get("policy_type"):     meta_bits.append(m["policy_type"])
        if m.get("account_type"):    meta_bits.append(m["account_type"])
        if m.get("purpose"):         meta_bits.append(m["purpose"])
        meta_str = (" [" + ",".join(meta_bits) + "]") if meta_bits else ""
        lines.append(f"{d['_id']} | {d.get('type')} | {d.get('label','')}{meta_str}")
    return "\n".join(lines)


GRAPH_CHAT_SYSTEM_PROMPT = """You are an expert financial-crime investigator embedded in a MongoDB-powered banking platform. Your knowledge source is the `banking.network_graph` collection — a unified graph of customers, accounts, loans, insurance policies, addresses, phones, devices, employers and counterparties.

You have ONE tool available, exposed as a JSON action you return:
  - explore_entity(entity_id, depth)  -- runs $graphLookup from a node, depth 1-3
  - none                              -- when the user asks a general question that needs no fresh data

GIVEN: the user's natural-language question, optional chat history, and a CATALOG of every node in the graph.

YOU MUST RETURN STRICT JSON (no markdown fences) with this shape:
{
  "narrative": "<concise plain-English answer in 2-5 sentences. Reference specific entities. Mention any fraud signals such as shared device, AML flag, ghost beneficiary, guarantor concentration. Do NOT use markdown headings or bullets.>",
  "action": {
    "type": "explore_entity" | "none",
    "entity_id": "<exact node _id from the catalog, ONLY if type=explore_entity>",
    "depth": <integer 1-3, default 2>
  },
  "highlight_nodes": [<list of node _ids worth flagging in the visualisation, can include entities not in the action>],
  "reasoning": "<one sentence explaining your tool choice — for transparency>"
}

RULES:
- The catalog will list every node as: ID | TYPE | LABEL [meta]. Use exact IDs from the left column.
- Pick depth=1 for "who owns X" questions; depth=2 for "tell me about X" or fraud-ring style questions; depth=3 only for transitive multi-hop AML traces.
- If the user names something vaguely (e.g. "the device fraud ring"), find the most likely matching node from the catalog and use its ID.
- If there is no good entity to explore (e.g. "what scenarios can I run?"), set action.type="none".
- Tone: confident, factual, no hedging. Use customer names from the label column rather than IDs in the narrative when natural."""


@app.post("/api/graph/chat")
async def api_graph_chat(
    request: Request,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    User asks a natural-language question. Claude picks a graph action
    and produces a narrative; backend executes the action and returns
    the combined response (graph data + narrative + pipeline used).
    """
    graph: Collection[Any] = request.app.state.network_graph
    llm_url: str = request.app.state.llm_url

    query: str = (payload or {}).get("query", "").strip()
    history: list[dict[str, str]] = (payload or {}).get("history", []) or []
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    t0 = time.time()
    catalog = _build_graph_catalog(graph)

    # Pre-detected alerts give the LLM ground-truth on known patterns so it
    # can answer "who is the ghost beneficiary?" or "show me the AML chain"
    # without having to infer from the catalog alone.
    alerts = _detect_risk_alerts(graph)
    alert_lines = []
    for a in alerts:
        alert_lines.append(
            f"- [{a['severity'].upper()}] {a['kind']}: {a['title']} "
            f"(focus_node={a.get('focus_node','?')}, summary={a.get('summary','')})"
        )
    alert_block = ("\n".join(alert_lines)) if alert_lines else "(none detected)"

    messages: list[dict[str, str]] = [
        {"role": "system",
         "content": (
             GRAPH_CHAT_SYSTEM_PROMPT
             + "\n\n--- CATALOG (every node) ---\n" + catalog
             + "\n\n--- DETECTED RISK PATTERNS (use these focus_node IDs when the user asks about a known pattern) ---\n"
             + alert_block
         )},
    ]
    # Replay last 6 turns of history for context
    for h in history[-6:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:2000]})
    messages.append({"role": "user", "content": query})

    # Call Claude
    try:
        resp = requests.post(llm_url, json={"messages": messages}, timeout=55)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    raw = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()

    # Tolerate markdown fences if the model wraps its JSON
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        # Graceful fallback — surface the raw text and skip the action
        return {
            "message": raw or "I could not parse a structured response.",
            "action": {"type": "none"},
            "highlight_nodes": [],
            "nodes": [], "edges": [],
            "pipeline_used": None,
            "query_time_ms": round((time.time() - t0) * 1000, 1),
        }

    narrative = str(plan.get("narrative", "")).strip()
    action = plan.get("action") or {"type": "none"}
    highlight_nodes = plan.get("highlight_nodes") or []
    reasoning = plan.get("reasoning", "")

    nodes_out: list[dict] = []
    edges_out: list[dict] = []
    pipeline_used: dict | None = None
    root_label = None
    root_type = None

    # Execute action
    if action.get("type") == "explore_entity":
        entity_id = action.get("entity_id")
        depth = max(1, min(3, int(action.get("depth", 2) or 2)))
        if entity_id:
            root_doc = graph.find_one({"_id": entity_id})
            if root_doc:
                pipeline = [
                    {"$match": {"_id": entity_id}},
                    {"$graphLookup": {
                        "from": GRAPH_COL_NAME,
                        "startWith": "$relationships.target",
                        "connectFromField": "relationships.target",
                        "connectToField": "_id",
                        "as": "connected",
                        "maxDepth": depth - 1,
                        "depthField": "hops",
                    }},
                ]
                result = list(graph.aggregate(pipeline))
                if result:
                    root = result[0]
                    connected = root.get("connected", [])
                    nodes_out = [_serialize_node(root, hops=0, is_root=True)]
                    seen = {root["_id"]}
                    for c in connected:
                        if c["_id"] in seen:
                            continue
                        nodes_out.append(_serialize_node(c, hops=int(c.get("hops", 0)) + 1))
                        seen.add(c["_id"])
                        if len(nodes_out) >= GRAPH_NODE_LIMIT:
                            break
                    for d in [root] + connected:
                        if d["_id"] not in seen:
                            continue
                        for e in _flatten_edges(d):
                            if e["to"] in seen:
                                edges_out.append(e)
                    pipeline_used = {
                        "description": f"$graphLookup from {entity_id} (depth={depth})",
                        "stages": pipeline,
                    }
                    root_label = root.get("label", entity_id)
                    root_type = root.get("type")

    return {
        "message": narrative,
        "reasoning": reasoning,
        "action": action,
        "highlight_nodes": highlight_nodes,
        "root": action.get("entity_id"),
        "root_label": root_label,
        "root_type": root_type,
        "nodes": nodes_out,
        "edges": edges_out,
        "pipeline_used": pipeline_used,
        "query_time_ms": round((time.time() - t0) * 1000, 1),
    }


# ──────────────────────────────────────────────────────────
# LangGraph-powered Investigator (v2)
# Five-node state machine (intent → resolver → planner → executor → narrator).
# Returns a full step trace so the UI can show the orchestration live.
# ──────────────────────────────────────────────────────────

@app.post("/api/graph/investigate")
async def api_graph_investigate(
    request: Request,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """LangGraph state machine over banking.network_graph."""
    from graph_agent import build_catalog, run_investigation

    graph: Collection[Any] = request.app.state.network_graph
    llm_url: str = request.app.state.llm_url

    query = (payload or {}).get("query", "").strip()
    history = (payload or {}).get("history", []) or []
    show_all = bool((payload or {}).get("show_all_relationships", False))
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    # Build prompt context once per request.
    catalog, catalog_ids = build_catalog(graph)
    alerts = _detect_risk_alerts(graph)

    try:
        result = run_investigation(
            query=query,
            history=history,
            graph=graph,
            db=graph.database,           # multi-collection provenance lookups
            llm_url=llm_url,
            catalog=catalog,
            catalog_ids=catalog_ids,
            alerts=alerts,
            show_all_relationships=show_all,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investigator error: {e}")

    # Serialize anything BSON-y the LangGraph may have produced.
    return _serialize(result)


# ──────────────────────────────────────────────────────────
# Document → Graph: extract entities from an onboarding document
# (text or PDF) and auto-link them into the network_graph collection.
# This is what makes the graph "live": as documents land in the bank,
# their entities are matched against existing nodes and any shared
# attribute immediately surfaces fraud-ring membership.
# ──────────────────────────────────────────────────────────

ENTITY_EXTRACT_PROMPT = """You are an entity-extraction system for a bank's compliance team.
Identify the document type and extract every relevant entity.

Return STRICT JSON (no markdown, no commentary), exactly this shape:
{
  "document_type": "national_id" | "passport" | "drivers_license" | "insurance_policy" | "kyc_form" | "bank_statement" | "other",
  "person":   { "name": "<full name or null>", "id_number": "<national id/passport/license number or null>", "dob": "<YYYY-MM-DD or null>", "nationality": "<or null>" },
  "addresses":[ { "text": "<full single-line address>" } ],
  "phones":   [ { "number": "<as written>" } ],
  "employers":[ { "name": "<company name>", "role": "<job title or null>" } ],
  "account_numbers":[ "<account string>" ],
  "policy": {
    "policy_id":          "<policy number as printed, or null>",
    "policy_type":        "life" | "health" | "auto" | "home" | "other" | null,
    "coverage_usd":       <number or null>,
    "premium_monthly_usd":<number or null>,
    "issue_date":         "<YYYY-MM-DD or null>",
    "expiry_date":        "<YYYY-MM-DD or null>",
    "underwriter":        "<insurer name or null>",
    "holder_name":        "<full name of policy holder or null>",
    "beneficiaries":      [ { "name": "<full name>", "relationship": "<spouse|child|parent|sibling|other or null>" } ]
  },
  "summary": "<one sentence describing the document>"
}

Rules:
- If a section does not apply (e.g. an ID card has no "policy" data) leave the inner fields null / [].
- If unsure, use null or empty array. Do NOT invent data.
- Preserve addresses exactly as written so they can be matched against records.
- For insurance_policy documents, holder_name and beneficiaries[] are the most important fields."""


def _normalize_addr(s: str) -> str:
    """Loose canonical form for address matching."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _normalize_phone(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _normalize_str(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _extract_entities_via_llm(llm_url: str, text_or_messages, is_image: bool = False) -> dict[str, Any]:
    """
    Invoke Claude via the LLM API gateway. `text_or_messages` may be a string
    (text mode) or a list of message blocks (multimodal).
    """
    if is_image:
        messages = text_or_messages
    else:
        messages = [
            {"role": "system", "content": ENTITY_EXTRACT_PROMPT},
            {"role": "user", "content": text_or_messages},
        ]
    try:
        resp = requests.post(llm_url, json={"messages": messages}, timeout=70)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"LLM extraction failed: {e}")
    raw = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw, "person": {"name": None}, "addresses": [], "phones": [],
                "employers": [], "account_numbers": [], "policy_numbers": [], "summary": ""}


def _match_extracted_entity(graph: Collection[Any], target_type: str, normalized_value: str,
                            label_field: str = "label") -> dict | None:
    """Find an existing node whose normalized label matches the extracted text.
    Kept for backwards compatibility — _match_with_activity is the new preferred call."""
    if not normalized_value:
        return None
    candidates = graph.find({"type": target_type}, {"_id": 1, "label": 1, "metadata": 1})
    for c in candidates:
        if target_type == "Address" and _normalize_addr(c.get("label", "")) == normalized_value:
            return c
        if target_type == "Phone" and _normalize_phone(c.get("label", "")) == normalized_value:
            return c
        if target_type == "Employer":
            existing = _normalize_str(c.get("label", ""))
            if existing == normalized_value or existing in normalized_value or normalized_value in existing:
                return c
    return None


def _normalize_label(s: str) -> str:
    """Mirror of setup_graph.normalize_label — must produce identical canonical form."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _match_with_activity(graph: Collection[Any], target_type: str, original_text: str
                         ) -> tuple[dict | None, list[dict]]:
    """
    Find an existing node and capture the live MongoDB pipeline as an Activity entry.

    MongoDB best practice: store a `normalized_label` field at WRITE time and do an
    indexed equality match at READ time — sub-millisecond, idiomatic, no $regex scan.
    The match + linked-customers fan-out is done in ONE aggregation using $lookup with
    a sub-pipeline (modern syntax, deterministic optimiser plan).
    """
    activities: list[dict] = []
    if not original_text or not original_text.strip():
        return None, activities

    normalized = _normalize_label(original_text)

    # ──────────────────────────────────────────────────────────
    # Single aggregation that:
    #   1. Indexed equality match on the (type, normalized_label) compound index
    #   2. $lookup with sub-pipeline to find every Customer linked back to this node
    #   3. $project a flat result with the count + linked customer summaries
    # ──────────────────────────────────────────────────────────
    pipeline = [
        # (1) Indexed equality match — uses the (type, normalized_label) compound index
        {"$match": {"type": target_type, "normalized_label": normalized}},

        # (2) Modern $lookup with sub-pipeline: pull every Customer that has a
        #     relationship pointing into this node. Avoids a second round-trip.
        {"$lookup": {
            "from": GRAPH_COL_NAME,
            "let": {"target_node_id": "$_id"},
            "pipeline": [
                {"$match": {"$expr": {"$and": [
                    {"$eq": ["$type", "Customer"]},
                    {"$in": ["$$target_node_id", {"$ifNull": ["$relationships.target", []]}]},
                ]}}},
                {"$project": {"_id": 1, "label": 1, "watchlist": "$metadata.watchlist"}},
            ],
            "as": "linked_customers",
        }},

        # (3) Reshape: surface the count + the customer list (preserve metadata.shared_by
        # so the downstream _classify call can read it as authoritative ground truth)
        {"$project": {
            "_id": 1, "type": 1, "label": 1,
            "metadata": 1,
            "shared_by_count": {"$size": "$linked_customers"},
            "linked_customers": 1,
        }},
    ]

    t0 = time.time()
    result = list(graph.aggregate(pipeline))
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    matched = result[0] if result else None
    linked = matched.get("linked_customers", []) if matched else []
    if matched:
        # Use the LIVE linked-customer count from this query as the authoritative
        # shared_by value — this stays in sync with the actual edges, beating the
        # static `metadata.shared_by` written at build time.
        matched.setdefault("metadata", {})
        matched["metadata"]["shared_by"] = max(
            len(linked), int(matched["metadata"].get("shared_by", 0) or 0)
        )

    # ── Surface the "non-PK indexed lookup" story to the UI ────────────
    # target_type tells us whether this match was on a primary-key-style
    # field (Customer name, Loan id) or a true non-PK attribute (Address,
    # Phone, Device, Employer). Non-PK matches are the killer-feature
    # demonstration: any string can be used as a key with the right index.
    non_pk_types = {"Address", "Phone", "Device", "Employer", "Counterparty"}
    is_non_pk = target_type in non_pk_types

    # Friendly SQL equivalent so the audience can compare directly.
    sql_contrast = {
        "would_require": [
            f"a `{target_type.lower()}s` master table",
            "a `customer_<x>` junction table per relationship type",
            "(and to match anything non-canonical) the `pg_trgm` extension or a GIN index",
        ],
        "example_sql": (
            f"-- PostgreSQL equivalent for matching an existing {target_type.lower()} "
            f"and listing every customer linked to it\n"
            f"SELECT a.id, a.label,\n"
            f"       array_agg(DISTINCT c.id) AS linked_customer_ids\n"
            f"FROM {target_type.lower()}s a\n"
            f"LEFT JOIN customer_{target_type.lower()}s ca ON ca.{target_type.lower()}_id = a.id\n"
            f"LEFT JOIN customers c ON c.id = ca.customer_id\n"
            f"WHERE a.normalized_label = lower(trim(:input_text))\n"
            f"GROUP BY a.id;\n"
            f"-- Requires: {target_type.lower()}s table + junction table + B-tree index\n"
            f"-- on normalized_label. Without normalization, falls back to ILIKE %x%\n"
            f"-- which cannot use a B-tree index → sequential scan."
        ),
        "key_difference": (
            "MongoDB stores relationships embedded inside the matched node "
            "(`relationships: []` array on every document). The `$lookup` sub-pipeline "
            "uses a multikey index on `relationships.target` to fan out to linked "
            "customers in the SAME aggregation — no junction table, no second round-trip. "
            "PostgreSQL forces a normalized 3-table model and 2-3 JOINs per match."
        ),
    }

    activities.append({
        "step": 1,
        "title": f"Match extracted {target_type} against the graph",
        "description": (
            f"One indexed-equality lookup followed by a $lookup sub-pipeline that "
            f"fans out to every customer linked back to the matched node. Uses the "
            f"compound index `(type, normalized_label)` for the match and "
            f"`relationships.target` for the join. Returns in a single round-trip."
        ),
        "operation": "aggregate",
        "collection": "banking.network_graph",
        "pipeline": pipeline,
        "execution_ms": elapsed_ms,
        "result_count": len(result),
        "shared_by_count": len(linked),
        "results": [_serialize(r) for r in result],
        "best_practice_note": (
            "❌ Avoided: case-insensitive $regex over the entire collection (collection scan).\n"
            "✅ Used: a `normalized_label` field stored at write time + compound index → "
            "sub-millisecond indexed equality even at billions of rows."
        ),

        # ── New: transparent provenance / "MongoDB Search Power" metadata ──
        "field_kind": "non_primary_key_string" if is_non_pk else "name_or_id",
        "target_type": target_type,
        "is_non_pk_match": is_non_pk,
        "index_used": {
            "name": "type_1_normalized_label_1",
            "keys": [["type", 1], ["normalized_label", 1]],
            "kind": "compound (B-tree)",
            "matched_on_field": "normalized_label",
            "complexity_note": (
                f"O(log N) lookup, ~{elapsed_ms} ms here against the live cluster — "
                f"stays O(log N) at 1M+ documents because the index is sorted on "
                f"(type, normalized_label) and we match both keys exactly."
            ),
        },
        "secondary_index_used": {
            "name": "relationships.target_1",
            "keys": [["relationships.target", 1]],
            "kind": "multikey",
            "purpose": "Fan-out from the matched attribute node to every Customer document whose embedded `relationships[].target` points back to it. Enables the JOIN-equivalent in a single aggregation.",
        },
        "sql_contrast": sql_contrast,
    })

    return matched, activities


def _beneficiary_policy_count_activity(graph: Collection[Any], beneficiary_id: str) -> dict:
    """Aggregation that counts how many existing policies a customer is beneficiary of.
    Surfaced to the UI as proof of the ghost-beneficiary check."""
    pipeline = [
        # Indexed equality on `type` plus an indexed lookup into `relationships.target`
        # via $elemMatch (the array index supports this).
        {"$match": {
            "type": "InsurancePolicy",
            "relationships": {"$elemMatch": {"target": beneficiary_id, "type": "BENEFICIARY_IS"}},
        }},
        {"$project": {
            "_id": 1, "label": 1,
            "metadata.coverage_usd": 1,
            "metadata.policy_type": 1,
            "metadata.underwriter": 1,
        }},
    ]
    t0 = time.time()
    results = list(graph.aggregate(pipeline))
    elapsed = round((time.time() - t0) * 1000, 1)
    return {
        "step": 1,
        "title": f"Count existing policies where {beneficiary_id} is the beneficiary",
        "description": (
            "$elemMatch over the `relationships` array — supported by the multi-key "
            "index on `relationships.target`. Every policy node that points to this "
            "customer with a BENEFICIARY_IS edge is returned in one pass. When count ≥3 "
            "we fire the ghost-beneficiary alert."
        ),
        "operation": "aggregate",
        "collection": "banking.network_graph",
        "pipeline": pipeline,
        "execution_ms": elapsed,
        "result_count": len(results),
        "results": [_serialize(r) for r in results],
    }


def _allocate_new_customer_id(graph: Collection[Any], proposed_nodes: list | None = None) -> str:
    """Allocate the next CUST-NEW-NNN id, accounting for nodes already allocated
    within the current request (not yet committed to the DB)."""
    in_db = graph.count_documents({"_id": {"$regex": "^CUST-NEW-"}})
    in_flight = sum(1 for n in (proposed_nodes or []) if str(n.get("_id", "")).startswith("CUST-NEW-"))
    return f"CUST-NEW-{in_db + in_flight + 1:03d}"


def _allocate_new_policy_id(graph: Collection[Any], proposed_nodes: list | None = None) -> str:
    in_db = graph.count_documents({"_id": {"$regex": "^POL-NEW-"}})
    in_flight = sum(1 for n in (proposed_nodes or []) if str(n.get("_id", "")).startswith("POL-NEW-"))
    return f"POL-NEW-{in_db + in_flight + 1:03d}"


def _match_customer_by_name(graph: Collection[Any], name: str) -> dict | None:
    """Indexed-equality lookup using the (type, normalized_label) compound index."""
    if not name:
        return None
    return graph.find_one(
        {"type": "Customer", "normalized_label": _normalize_label(name)},
        {"_id": 1, "label": 1, "metadata": 1},
    )


def _customer_match_activity(graph: Collection[Any], name: str) -> tuple[dict | None, dict]:
    """Same as _match_customer_by_name but also captures the query as an activity entry."""
    normalized = _normalize_label(name)
    query = {"type": "Customer", "normalized_label": normalized}
    t0 = time.time()
    matched = graph.find_one(query, {"_id": 1, "label": 1, "metadata": 1})
    elapsed = round((time.time() - t0) * 1000, 1)
    return matched, {
        "step": 1,
        "title": f"Resolve \"{name}\" to an existing customer",
        "description": (
            "Compound-index equality lookup `(type, normalized_label)`. "
            "We store a pre-canonicalised label at write time so name matching is "
            "always indexed equality — never a collection-scanning regex."
        ),
        "operation": "findOne",
        "collection": "banking.network_graph",
        "query": query,
        "execution_ms": elapsed,
        "result_count": 1 if matched else 0,
        "results": [_serialize(matched)] if matched else [],

        # Make the index transparent for the audience.
        "field_kind": "name_or_id",
        "target_type": "Customer",
        "is_non_pk_match": False,
        "index_used": {
            "name": "type_1_normalized_label_1",
            "keys": [["type", 1], ["normalized_label", 1]],
            "kind": "compound (B-tree)",
            "matched_on_field": "normalized_label",
            "complexity_note": (
                f"O(log N) lookup, ~{elapsed} ms here. The same index that powers "
                f"non-PK matches (address/phone/device) also powers name resolution — "
                f"one index covers every string-equality lookup the demo needs."
            ),
        },
    }


def _count_existing_beneficiary_policies(graph: Collection[Any], cust_id: str) -> int:
    """Count InsurancePolicy nodes where this customer is a BENEFICIARY_IS target."""
    return graph.count_documents({
        "type": "InsurancePolicy",
        "relationships": {"$elemMatch": {"target": cust_id, "type": "BENEFICIARY_IS"}},
    })


@app.post("/api/graph/extract")
async def api_graph_extract(
    request: Request,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Accept an onboarding document as TEXT (JSON body: {text, commit}) and
    extract entities via Claude, match them against the existing graph,
    optionally insert new nodes/edges. Returns a full extraction report
    including risk alerts triggered by matches against shared identifiers.
    """
    graph: Collection[Any] = request.app.state.network_graph
    llm_url: str = request.app.state.llm_url

    payload = payload or {}
    text: str = (payload.get("text") or "").strip()
    commit: bool = bool(payload.get("commit", True))
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > 6000:
        raise HTTPException(status_code=400, detail="text too long (max 6000 chars)")

    return _run_extraction(graph, llm_url, text=text, source_label="text_paste",
                           commit=commit, image_messages=None)


@app.post("/api/graph/extract-pdf")
async def api_graph_extract_pdf(
    request: Request,
    file: UploadFile = File(...),
    commit: bool = Query(True),
) -> dict[str, Any]:
    """Same as /api/graph/extract but accepts a PDF/image upload (Bedrock multimodal)."""
    graph: Collection[Any] = request.app.state.network_graph
    llm_url: str = request.app.state.llm_url

    content_type = (file.content_type or "").lower()
    ext = (file.filename or "").rsplit(".", 1)[-1].lower() if file.filename else ""
    ext_to_mime = {"pdf": "application/pdf", "png": "image/png",
                   "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
    if content_type not in ALLOWED_UPLOAD_TYPES:
        content_type = ext_to_mime.get(ext, content_type)
    if content_type not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, PNG, JPG, WEBP")

    file_bytes = await file.read()
    if len(file_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 8 MB)")

    file_b64 = base64.b64encode(file_bytes).decode("utf-8")
    block_type = ALLOWED_UPLOAD_TYPES[content_type]
    media_block = {
        "type": "document" if block_type == "document" else "image",
        "source": {"type": "base64", "media_type": content_type, "data": file_b64},
    }
    messages = [
        {"role": "system", "content": ENTITY_EXTRACT_PROMPT},
        {"role": "user", "content": [media_block, {"type": "text", "text": "Extract all entities from this document."}]},
    ]

    return _run_extraction(graph, llm_url, text=None, source_label=f"upload:{file.filename}",
                           commit=commit, image_messages=messages)


def _extract_insurance_policy(
    graph: Collection[Any],
    extraction: dict,
    policy: dict,
    source_label: str,
    commit: bool,
    t0: float,
) -> dict[str, Any]:
    """
    Process an insurance-policy document. Creates an InsurancePolicy node,
    matches/creates the holder Customer, matches/creates each beneficiary
    Customer, and fires a ghost-beneficiary alert if any beneficiary is now
    named on >= 3 policies.
    """
    matches: list[dict] = []
    proposed_nodes: list[dict] = []
    proposed_edges: list[dict] = []
    risk_alerts: list[dict] = []

    holder_name = (policy.get("holder_name") or "").strip()
    if not holder_name:
        return {
            "error": "Insurance policy has no holder_name — cannot link.",
            "extraction": extraction,
            "query_time_ms": round((time.time() - t0) * 1000, 1),
        }

    new_pol_id = _allocate_new_policy_id(graph, proposed_nodes)
    coverage = policy.get("coverage_usd")
    pol_type = (policy.get("policy_type") or "").lower() or "life"
    label = f"{pol_type.title()} Policy ${int(coverage):,}" if coverage else f"{pol_type.title()} Policy"

    proposed_nodes.append({
        "_id": new_pol_id,
        "type": "InsurancePolicy",
        "label": label,
        "metadata": {
            "policy_id_printed": policy.get("policy_id"),
            "policy_type": pol_type,
            "coverage_usd": coverage,
            "premium_monthly_usd": policy.get("premium_monthly_usd"),
            "underwriter": policy.get("underwriter"),
            "issue_date": policy.get("issue_date"),
            "expiry_date": policy.get("expiry_date"),
            "onboarding_source": source_label,
            "auto_extracted": True,
        },
    })

    # ── Holder matching ──
    src_h = f"extracted from {source_label} (policy holder field)"
    holder_match, holder_activity = _customer_match_activity(graph, holder_name)
    holder_record = {
        "kind": "PolicyHolder", "extracted": holder_name,
        "activities": [holder_activity],
    }
    if holder_match:
        holder_id = holder_match["_id"]
        on_watchlist = (holder_match.get("metadata") or {}).get("watchlist", False)
        holder_record.update({
            "matched_id": holder_id, "matched_label": holder_match["label"],
            "previously_shared_by": 0,
            "match_type": "watchlist" if on_watchlist else "identity",
            "reason": (
                f"Policy holder name matches existing customer {holder_id}. "
                + ("⚠ This customer is currently on the watchlist." if on_watchlist
                   else "Linking the new policy to the existing customer record.")
            ),
        })
        matches.append(holder_record)
        if on_watchlist:
            risk_alerts.append({
                "severity": "high", "kind": "watchlist_holder",
                "message": f"⚠ Policy holder \"{holder_name}\" is on the watchlist.",
                "matched_node": holder_id,
            })
    else:
        holder_record.update({
            "match_type": "new",
            "reason": "Holder name not on file — creating a new customer record alongside the policy.",
        })
        matches.append(holder_record)
        holder_id = _allocate_new_customer_id(graph, proposed_nodes)
        proposed_nodes.append({
            "_id": holder_id, "type": "Customer", "label": holder_name,
            "metadata": {
                "onboarding_source": source_label,
                "auto_extracted": True,
                "watchlist": False,
                "via_policy_doc": True,
            },
        })
    proposed_edges.append({"from": holder_id, "to": new_pol_id, "type": "POLICY_HOLDER", "source": src_h})
    proposed_edges.append({"from": new_pol_id, "to": holder_id, "type": "HELD_BY", "source": src_h})

    # ── Beneficiary matching ──
    src_b = f"extracted from {source_label} (beneficiary list)"
    for b in policy.get("beneficiaries", []) or []:
        bname = (b.get("name") or "").strip()
        if not bname:
            continue
        b_match, b_activity = _customer_match_activity(graph, bname)
        b_record = {"kind": "Beneficiary", "extracted": bname, "activities": [b_activity]}
        if b_match:
            b_id = b_match["_id"]
            ghost_activity = _beneficiary_policy_count_activity(graph, b_id)
            b_record["activities"].append(ghost_activity)
            existing_count = ghost_activity["result_count"]
            new_count = existing_count + 1
            on_watchlist = (b_match.get("metadata") or {}).get("watchlist", False)
            # Ghost-beneficiary alert if total reaches >=3
            ghost_match = new_count >= 3
            if ghost_match:
                match_type = "fraud_signal"
                reason = (
                    f"⚠ Beneficiary {bname} is already named on {existing_count} other policies — "
                    f"adding this would make {new_count} total. Ghost-beneficiary pattern is "
                    f"flagged when one person is beneficiary of multiple unrelated holders."
                )
            elif on_watchlist:
                match_type = "watchlist"
                reason = f"Beneficiary {bname} is on the watchlist (currently named on {existing_count} policies)."
            else:
                match_type = "identity"
                reason = (
                    f"Beneficiary {bname} matches existing customer {b_id}. "
                    f"Currently named on {existing_count} other policies "
                    f"(below the {3}-policy ghost-beneficiary threshold)."
                )
            b_record.update({
                "matched_id": b_id, "matched_label": b_match["label"],
                "previously_shared_by": existing_count,
                "match_type": match_type, "reason": reason,
            })
            matches.append(b_record)
            if ghost_match:
                severity = "high" if new_count >= 5 else "medium"
                risk_alerts.append({
                    "severity": severity, "kind": "ghost_beneficiary",
                    "message": (
                        f"⚠ \"{bname}\" is now named as beneficiary on {new_count} policies "
                        f"(was {existing_count}). Ghost beneficiary pattern."
                    ),
                    "matched_node": b_id,
                })
            if on_watchlist:
                risk_alerts.append({
                    "severity": "high", "kind": "watchlist_beneficiary",
                    "message": f"⚠ Beneficiary \"{bname}\" is on the watchlist.",
                    "matched_node": b_id,
                })
        else:
            b_record.update({
                "match_type": "new",
                "reason": "Beneficiary name not on file — creating a new customer record.",
            })
            matches.append(b_record)
            b_id = _allocate_new_customer_id(graph, proposed_nodes)
            proposed_nodes.append({
                "_id": b_id, "type": "Customer", "label": bname,
                "metadata": {
                    "onboarding_source": source_label,
                    "auto_extracted": True,
                    "watchlist": False,
                    "via_policy_doc": True,
                    "relationship": b.get("relationship"),
                },
            })
        proposed_edges.append({"from": b_id, "to": new_pol_id, "type": "BENEFICIARY_OF", "source": src_b})
        proposed_edges.append({"from": new_pol_id, "to": b_id, "type": "BENEFICIARY_IS", "source": src_b})

    # ── Commit ──
    inserted_ids: list[str] = []
    if commit and proposed_nodes:
        edges_by_src: dict[str, list[dict]] = {}
        for e in proposed_edges:
            edges_by_src.setdefault(e["from"], []).append({
                "target": e["to"], "type": e["type"], "source": e.get("source", ""),
            })
        existing_ids = {d["_id"] for d in graph.find(
            {"_id": {"$in": [n["_id"] for n in proposed_nodes]}}, {"_id": 1})}
        new_docs = []
        for n in proposed_nodes:
            if n["_id"] in existing_ids:
                continue
            new_docs.append({
                "_id": n["_id"], "type": n["type"], "label": n["label"],
                "metadata": n.get("metadata", {}),
                "relationships": edges_by_src.get(n["_id"], []),
                "source_documents": [source_label],
            })
            inserted_ids.append(n["_id"])
        if new_docs:
            graph.insert_many(new_docs)
        # Append edges to existing nodes
        for src_id, rels in edges_by_src.items():
            if src_id in inserted_ids:
                continue
            for r in rels:
                graph.update_one(
                    {"_id": src_id,
                     "relationships": {"$not": {"$elemMatch": {"target": r["target"], "type": r["type"]}}}},
                    {"$push": {"relationships": r}}
                )

    # ── Build subgraph centered on the new policy ──
    nodes_out: list[dict] = []
    edges_out: list[dict] = []
    if commit and inserted_ids:
        result = list(graph.aggregate([
            {"$match": {"_id": new_pol_id}},
            {"$graphLookup": {
                "from": GRAPH_COL_NAME,
                "startWith": "$relationships.target",
                "connectFromField": "relationships.target",
                "connectToField": "_id",
                "as": "connected", "maxDepth": 1, "depthField": "hops",
            }},
        ]))
        if result:
            root = result[0]
            connected = root.get("connected", [])
            nodes_out = [_serialize_node(root, hops=0, is_root=True)]
            seen = {root["_id"]}
            for c in connected:
                if c["_id"] in seen:
                    continue
                nodes_out.append(_serialize_node(c, hops=int(c.get("hops", 0)) + 1))
                seen.add(c["_id"])
            for d in [root] + connected:
                if d["_id"] not in seen:
                    continue
                for e in _flatten_edges(d):
                    if e["to"] in seen:
                        edges_out.append(e)

    return {
        "extraction": extraction,
        "document_type": "insurance_policy",
        "new_policy_id": new_pol_id,
        "policy": {
            "holder_id": holder_id,
            "holder_name": holder_name,
            "beneficiaries": policy.get("beneficiaries", []),
            "coverage_usd": coverage,
            "policy_type": pol_type,
        },
        "matches": matches,
        "proposed_nodes": proposed_nodes,
        "proposed_edges": proposed_edges,
        "risk_alerts": risk_alerts,
        "committed": bool(commit and inserted_ids),
        "inserted_node_ids": inserted_ids,
        "highlight_nodes": [new_pol_id] + [m["matched_id"] for m in matches if m.get("matched_id")],
        "nodes": nodes_out,
        "edges": edges_out,
        "query_time_ms": round((time.time() - t0) * 1000, 1),
    }


def _run_extraction(graph: Collection[Any], llm_url: str, text: str | None,
                    source_label: str, commit: bool,
                    image_messages: list | None) -> dict[str, Any]:
    """Shared extraction + matching + commit logic for text & PDF endpoints."""
    t0 = time.time()

    # ── 1. LLM extraction ──
    if image_messages:
        extraction = _extract_entities_via_llm(llm_url, image_messages, is_image=True)
    else:
        extraction = _extract_entities_via_llm(llm_url, text)

    document_type = (extraction.get("document_type") or "").strip().lower()
    person = extraction.get("person") or {}
    addresses = extraction.get("addresses") or []
    phones = extraction.get("phones") or []
    employers = extraction.get("employers") or []
    account_numbers = extraction.get("account_numbers") or []
    policy = extraction.get("policy") or {}
    summary = extraction.get("summary") or ""

    matches: list[dict] = []
    proposed_nodes: list[dict] = []
    proposed_edges: list[dict] = []
    risk_alerts: list[dict] = []

    # ──────────────────────────────────────────────────────────
    # BRANCH A: Insurance Policy → create Policy node + holder/beneficiary edges
    # ──────────────────────────────────────────────────────────
    if document_type == "insurance_policy" or (policy.get("holder_name") and policy.get("beneficiaries")):
        return _extract_insurance_policy(
            graph, extraction, policy, source_label, commit, t0,
        )

    # ──────────────────────────────────────────────────────────
    # BRANCH B: KYC / national_id / passport → onboard a person
    # ──────────────────────────────────────────────────────────

    # 2a. Person — allocate a new customer
    person_name = (person.get("name") or "").strip()
    if not person_name:
        return {
            "error": "No person name extracted. Cannot onboard without an identity.",
            "extraction": extraction,
            "query_time_ms": round((time.time() - t0) * 1000, 1),
        }
    new_cust_id = _allocate_new_customer_id(graph, proposed_nodes)
    proposed_nodes.append({
        "_id": new_cust_id,
        "type": "Customer",
        "label": person_name,
        "metadata": {
            "id_number": person.get("id_number"),
            "dob": person.get("dob"),
            "nationality": person.get("nationality"),
            "document_type": document_type or "kyc_form",
            "onboarding_source": source_label,
            "watchlist": False,
            "auto_extracted": True,
        },
    })

    # ── helper to classify each match ────────────────────
    def _classify(existing_node: dict, kind: str) -> tuple[str, str]:
        """
        Returns (match_type, reason). match_type ∈
          - 'fraud_signal' : matched a seed-loaded node already shared by ≥2 customers
          - 'emerging'     : matched an auto-extracted node from a prior upload (informational)
          - 'identity'     : same self-match (rare edge case)
        """
        meta = existing_node.get("metadata") or {}
        shared_by = int(meta.get("shared_by", 0) or 0)
        is_auto = bool(meta.get("auto_extracted"))
        if not is_auto and shared_by >= 2:
            return ("fraud_signal",
                    f"This {kind.lower()} is on file for {shared_by} existing customer{'s' if shared_by != 1 else ''} "
                    f"in the bank's records — a known shared identifier flagged in the fraud-ring seed data.")
        if is_auto:
            return ("emerging",
                    f"This {kind.lower()} was first seen during a previous upload in this session "
                    f"(linked to {shared_by} prior customer{'s' if shared_by != 1 else ''}). "
                    f"Not a known fraud signal yet — informational only.")
        return ("identity",
                f"This {kind.lower()} matches a single seed-loaded record; no shared-identifier pattern detected.")

    # 2b. Address matching (with live MongoDB activity)
    for addr in addresses:
        text_addr = addr.get("text") or ""
        if not text_addr.strip():
            continue
        existing, activities = _match_with_activity(graph, "Address", text_addr)
        src = f"extracted from {source_label}"
        match_record: dict[str, Any] = {"kind": "Address", "extracted": text_addr, "activities": activities}
        if existing:
            match_type, reason = _classify(existing, "Address")
            shared_by = (existing.get("metadata") or {}).get("shared_by", 0)
            match_record.update({
                "matched_id": existing["_id"], "matched_label": existing["label"],
                "previously_shared_by": shared_by,
                "match_type": match_type, "reason": reason,
            })
            proposed_edges.append({"from": new_cust_id, "to": existing["_id"], "type": "LIVES_AT", "source": src})
            proposed_edges.append({"from": existing["_id"], "to": new_cust_id, "type": "RESIDENT", "source": src})
            if match_type == "fraud_signal":
                risk_alerts.append({
                    "severity": "high", "kind": "address_match_to_ring",
                    "message": (
                        f"⚠ Address \"{existing['label']}\" already linked to "
                        f"{shared_by} customers — onboarding this customer would make them "
                        f"member #{shared_by + 1} of a known synthetic-identity ring."
                    ),
                    "matched_node": existing["_id"],
                })
        else:
            match_record.update({"match_type": "new",
                                 "reason": "Address not seen before; a new Address node is inserted."})
            new_id = f"ADDR-EXTRACT-{int(time.time())%100000}-{len(proposed_nodes)}"
            proposed_nodes.append({"_id": new_id, "type": "Address", "label": text_addr,
                                   "metadata": {"shared_by": 1, "auto_extracted": True}})
            proposed_edges.append({"from": new_cust_id, "to": new_id, "type": "LIVES_AT", "source": src})
            proposed_edges.append({"from": new_id, "to": new_cust_id, "type": "RESIDENT", "source": src})
        matches.append(match_record)

    # 2c. Phone matching (with live MongoDB activity)
    for ph in phones:
        text_ph = ph.get("number") or ""
        if not text_ph.strip():
            continue
        existing, activities = _match_with_activity(graph, "Phone", text_ph)
        src = f"extracted from {source_label}"
        match_record = {"kind": "Phone", "extracted": text_ph, "activities": activities}
        if existing:
            match_type, reason = _classify(existing, "Phone")
            shared_by = (existing.get("metadata") or {}).get("shared_by", 0)
            match_record.update({
                "matched_id": existing["_id"], "matched_label": existing["label"],
                "previously_shared_by": shared_by,
                "match_type": match_type, "reason": reason,
            })
            proposed_edges.append({"from": new_cust_id, "to": existing["_id"], "type": "USES_PHONE", "source": src})
            proposed_edges.append({"from": existing["_id"], "to": new_cust_id, "type": "PHONE_OF", "source": src})
            if match_type == "fraud_signal":
                risk_alerts.append({
                    "severity": "high", "kind": "phone_match_to_ring",
                    "message": f"⚠ Phone {existing['label']} already linked to {shared_by} customers — fraud signal.",
                    "matched_node": existing["_id"],
                })
        else:
            match_record.update({"match_type": "new",
                                 "reason": "Phone number not seen before; a new Phone node is inserted."})
            new_id = f"PHONE-EXTRACT-{int(time.time())%100000}-{len(proposed_nodes)}"
            proposed_nodes.append({"_id": new_id, "type": "Phone", "label": text_ph,
                                   "metadata": {"shared_by": 1, "auto_extracted": True}})
            proposed_edges.append({"from": new_cust_id, "to": new_id, "type": "USES_PHONE", "source": src})
            proposed_edges.append({"from": new_id, "to": new_cust_id, "type": "PHONE_OF", "source": src})
        matches.append(match_record)

    # 2d. Employer matching (with live MongoDB activity)
    for emp in employers:
        text_emp = emp.get("name") or ""
        if not text_emp.strip():
            continue
        existing, activities = _match_with_activity(graph, "Employer", text_emp)
        src = f"extracted from {source_label}"
        match_record = {"kind": "Employer", "extracted": text_emp, "activities": activities}
        if existing:
            match_type, reason = _classify(existing, "Employer")
            shared_by = (existing.get("metadata") or {}).get("shared_by", 0)
            label_lower = existing["label"].lower()
            is_shell_word = any(k in label_lower for k in ["phantom", "shell", "ghost"])
            if match_type == "fraud_signal" and not is_shell_word:
                match_type = "emerging"
                reason = (f"This employer is on file for {shared_by} existing customers — "
                          f"likely a legitimate large employer rather than a fraud signal.")
            match_record.update({
                "matched_id": existing["_id"], "matched_label": existing["label"],
                "previously_shared_by": shared_by,
                "match_type": match_type, "reason": reason,
            })
            proposed_edges.append({"from": new_cust_id, "to": existing["_id"], "type": "WORKS_AT", "source": src})
            proposed_edges.append({"from": existing["_id"], "to": new_cust_id, "type": "EMPLOYS", "source": src})
            if match_type == "fraud_signal" and is_shell_word:
                risk_alerts.append({
                    "severity": "high", "kind": "employer_match_to_ring",
                    "message": f"⚠ Employer \"{existing['label']}\" is a known shell company in a fraud ring.",
                    "matched_node": existing["_id"],
                })
        else:
            match_record.update({"match_type": "new",
                                 "reason": "Employer not seen before; a new Employer node is inserted."})
            new_id = f"EMP-EXTRACT-{int(time.time())%100000}-{len(proposed_nodes)}"
            proposed_nodes.append({"_id": new_id, "type": "Employer", "label": text_emp,
                                   "metadata": {"shared_by": 1, "auto_extracted": True}})
            proposed_edges.append({"from": new_cust_id, "to": new_id, "type": "WORKS_AT", "source": src})
            proposed_edges.append({"from": new_id, "to": new_cust_id, "type": "EMPLOYS", "source": src})
        matches.append(match_record)

    # ── 3. Commit (if requested) ──
    inserted_node_ids: list[str] = []
    if commit and proposed_nodes:
        # Group edges by source node
        edges_by_src: dict[str, list[dict]] = {}
        for e in proposed_edges:
            edges_by_src.setdefault(e["from"], []).append({
                "target": e["to"],
                "type": e["type"],
                "source": e.get("source", ""),
            })

        # Insert NEW nodes only (those not already in graph)
        existing_ids = {d["_id"] for d in graph.find(
            {"_id": {"$in": [n["_id"] for n in proposed_nodes]}}, {"_id": 1})}
        new_docs = []
        for n in proposed_nodes:
            if n["_id"] in existing_ids:
                continue
            doc = {
                "_id": n["_id"],
                "type": n["type"],
                "label": n["label"],
                "metadata": n.get("metadata", {}),
                "relationships": edges_by_src.get(n["_id"], []),
                "source_documents": [source_label],
            }
            new_docs.append(doc)
            inserted_node_ids.append(n["_id"])
        if new_docs:
            graph.insert_many(new_docs)

        # Append edges to EXISTING nodes (matched targets) using $push
        for src_id, rels in edges_by_src.items():
            if src_id in inserted_node_ids:
                continue  # already written above
            # only push relationships that don't already exist (target+type)
            for r in rels:
                graph.update_one(
                    {"_id": src_id,
                     "relationships": {"$not": {"$elemMatch": {"target": r["target"], "type": r["type"]}}}},
                    {"$push": {"relationships": r}}
                )

        # If risk alerts were triggered, also flag the new customer with watchlist
        if risk_alerts:
            graph.update_one(
                {"_id": new_cust_id},
                {"$set": {"metadata.watchlist": True, "metadata.flagged_reason": "auto_match_fraud_ring"}}
            )

    # ── 4. Build a small subgraph centered on the new customer for visualization ──
    nodes_out: list[dict] = []
    edges_out: list[dict] = []
    if commit and inserted_node_ids:
        center_doc = graph.find_one({"_id": new_cust_id})
        if center_doc:
            pipeline = [
                {"$match": {"_id": new_cust_id}},
                {"$graphLookup": {
                    "from": GRAPH_COL_NAME,
                    "startWith": "$relationships.target",
                    "connectFromField": "relationships.target",
                    "connectToField": "_id",
                    "as": "connected",
                    "maxDepth": 1,
                    "depthField": "hops",
                }},
            ]
            result = list(graph.aggregate(pipeline))
            if result:
                root = result[0]
                connected = root.get("connected", [])
                nodes_out = [_serialize_node(root, hops=0, is_root=True)]
                seen = {root["_id"]}
                for c in connected:
                    if c["_id"] in seen:
                        continue
                    nodes_out.append(_serialize_node(c, hops=int(c.get("hops", 0)) + 1))
                    seen.add(c["_id"])
                for d in [root] + connected:
                    if d["_id"] not in seen:
                        continue
                    for e in _flatten_edges(d):
                        if e["to"] in seen:
                            edges_out.append(e)

    return {
        "extraction": {
            "person": person,
            "addresses": addresses,
            "phones": phones,
            "employers": employers,
            "account_numbers": account_numbers,
            "summary": summary,
        },
        "new_customer_id": new_cust_id,
        "matches": matches,
        "proposed_nodes": proposed_nodes,
        "proposed_edges": proposed_edges,
        "risk_alerts": risk_alerts,
        "committed": bool(commit and inserted_node_ids),
        "inserted_node_ids": inserted_node_ids,
        "highlight_nodes": [new_cust_id] + [m["matched_id"] for m in matches if m.get("matched_id")],
        "nodes": nodes_out,
        "edges": edges_out,
        "query_time_ms": round((time.time() - t0) * 1000, 1),
    }


@app.post("/api/graph/reset-extractions")
async def api_graph_reset_extractions(request: Request) -> dict[str, Any]:
    """Remove all auto-extracted nodes (CUST-NEW-*, ADDR-EXTRACT-*, etc.)."""
    graph: Collection[Any] = request.app.state.network_graph
    extracted_ids = [d["_id"] for d in graph.find(
        {"metadata.auto_extracted": True}, {"_id": 1})]
    if not extracted_ids:
        return {"removed_count": 0}
    # Remove the nodes themselves
    graph.delete_many({"_id": {"$in": extracted_ids}})
    # Also $pull any references to them from remaining nodes' relationships
    graph.update_many(
        {"relationships.target": {"$in": extracted_ids}},
        {"$pull": {"relationships": {"target": {"$in": extracted_ids}}}}
    )
    return {"removed_count": len(extracted_ids), "removed_ids": extracted_ids}


# ──────────────────────────────────────────────────────────
# Ask Your Data: NL → MongoDB aggregation → results
# ──────────────────────────────────────────────────────────

SCHEMA_CONTEXT = """
MongoDB database: "banking"

Collections and their fields:

1. customers:
   - _id (ObjectId), full_name, email, phone, date_of_birth, address, city, country
   - employment_type (salaried|self_employed|government|retired|student)
   - monthly_income (number), credit_score (number, 300-850)
   - payment_history (excellent|good|fair|poor)
   - account_age_months (number), financial_goals (string)

2. loan_applications:
   - _id (ObjectId), customer_id (ObjectId ref→customers)
   - loan_amount (number), term_months (number), interest_rate (number)
   - monthly_payment (number), purpose (string, e.g. "home","auto","education","personal","business")
   - status (approved|pending|declined)
   - application_date (Date)

3. accounts:
   - _id (ObjectId), customer_id (ObjectId ref→customers)
   - account_type (savings|checking), balance (number), currency (string)

4. transactions:
   - _id (ObjectId), account_id (ObjectId ref→accounts), customer_id (ObjectId ref→customers)
   - amount (number), type (credit|debit), category (string), description (string)
   - date (Date)

5. kyc_documents:
   - _id (ObjectId), customer_id (string), document_type (national_id|passport|drivers_license)
   - document_number, issue_date, expiry_date, verification_status (pending|verified|rejected)
   - risk_flags (array of strings), description (string)
   - description_embedding (1024-dim vector, Voyage AI voyage-4-large)
   - VECTOR INDEX: "kyc_vector_index" on field "description_embedding"

6. bank_products:
   - _id (ObjectId), name, description, category, features (array)
   - embedding (1024-dim vector, Voyage AI voyage-4-large)
   - VECTOR INDEX: "product_vector_index" on field "embedding"

7. faq_chunks:
   - _id (ObjectId), title, content_en, content_km, category
   - embedding (1024-dim vector, Voyage AI voyage-4-large)
   - VECTOR INDEX: "faq_vector_index" on field "embedding"

8. loan_support_docs:
   - _id (ObjectId), customer_id (ObjectId), filename, content_type
   - document_type, extracted_fields (object), full_text, summary, confidence
   - embedding (1024-dim vector, Voyage AI voyage-4-large) — may not exist on all docs
"""

ASK_DATA_SYSTEM_PROMPT = (
    "You are a MongoDB analytics expert for a banking application.\n"
    "Given the database schema below and a user's natural language question, "
    "generate a MongoDB aggregation pipeline that answers the question.\n\n"
    + SCHEMA_CONTEXT +
    "\n\nYou can use TWO types of queries:\n\n"
    "TYPE 1 — Standard Aggregation (for analytical/counting/filtering questions):\n"
    "Return: {\"collection\": \"<name>\", \"pipeline\": [<stages>]}\n"
    "Use operators like $match, $group, $sort, $project, $lookup, $unwind, $addFields, $bucket, $limit.\n\n"
    "TYPE 2 — Vector Search (for semantic/similarity/meaning-based questions):\n"
    "Return: {\"collection\": \"<name>\", \"pipeline\": [<stages>], \"vector_search\": {\"search_text\": \"<the text to search for>\", \"index\": \"<index_name>\", \"path\": \"<embedding_field>\", \"limit\": <n>}}\n"
    "When vector_search is present, the backend will:\n"
    "  1. Generate a 1024-dim embedding from search_text using Voyage AI\n"
    "  2. Prepend a $vectorSearch stage to the pipeline using the real vector\n"
    "  3. Execute the full pipeline\n"
    "Use this for questions like 'find FAQs about X', 'find products similar to Y', 'search documents about Z'.\n"
    "The pipeline stages you provide will run AFTER the $vectorSearch stage.\n"
    "Always add {\"$addFields\": {\"score\": {\"$meta\": \"vectorSearchScore\"}}} as your first pipeline stage.\n\n"
    "Rules:\n"
    "- Return ONLY valid JSON (no markdown fences)\n"
    "- For ObjectId references, use string comparison where needed\n"
    "- Limit results to 20 rows max\n"
    "- Always produce human-readable field names in output\n"
    "- If the question cannot be answered, return {\"error\": \"<reason>\"}\n"
    "- Decide between Type 1 and Type 2 based on whether the question needs semantic understanding or exact filtering/aggregation\n"
)


@app.post("/api/ask-data")
async def api_ask_data(
    request: Request,
    q: str = Query(..., min_length=3),
) -> dict[str, Any]:
    """
    Accept a natural language question, use Claude to generate a MongoDB
    aggregation pipeline, execute it, and return results + the generated query.
    """
    llm_url: str = request.app.state.llm_url
    db = request.app.state.db

    messages = [
        {"role": "system", "content": ASK_DATA_SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]

    t0 = time.time()
    try:
        resp = requests.post(llm_url, json={"messages": messages}, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
    llm_ms = round((time.time() - t0) * 1000)

    data = resp.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=f"LLM error: {data['error']}")
    raw = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    usage = data.get("usage", {})

    parsed: dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].rsplit("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].rsplit("```", 1)[0]
        try:
            parsed = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            return {"query": q, "error": "Could not parse LLM response as JSON", "raw_response": raw[:2000]}

    if "error" in parsed:
        return {"query": q, "error": parsed["error"], "llm_latency_ms": llm_ms}

    collection_name = parsed.get("collection", "")
    pipeline = parsed.get("pipeline", [])
    vector_search_spec = parsed.get("vector_search")

    if collection_name not in ("customers", "loan_applications", "accounts",
                                "transactions", "kyc_documents", "bank_products",
                                "loan_support_docs", "faq_chunks"):
        return {"query": q, "error": f"Unknown collection: {collection_name}"}

    # If Claude requested vector search, generate embedding and prepend $vectorSearch
    embed_ms = 0
    vector_search_meta: dict[str, Any] = {}
    if vector_search_spec and isinstance(vector_search_spec, dict):
        voyage_client: voyageai.Client = request.app.state.voyage
        search_text = vector_search_spec.get("search_text", q)
        index_name = vector_search_spec.get("index", "")
        path = vector_search_spec.get("path", EMBEDDING_FIELD)
        vs_limit = int(vector_search_spec.get("limit", 10))

        t_embed = time.time()
        query_vector = get_query_embedding(voyage_client, search_text)
        embed_ms = round((time.time() - t_embed) * 1000)

        vs_stage = {
            "$vectorSearch": {
                "index": index_name,
                "path": path,
                "queryVector": query_vector,
                "numCandidates": min(200, max(50, vs_limit * 20)),
                "limit": vs_limit,
            }
        }
        pipeline = [vs_stage] + pipeline

        vector_search_meta = {
            "search_text": search_text,
            "index": index_name,
            "path": path,
            "embedding_model": VOYAGE_EMBED_MODEL,
            "dimensions": len(query_vector),
            "embed_latency_ms": embed_ms,
        }

    t1 = time.time()
    try:
        results = list(db[collection_name].aggregate(pipeline))
    except Exception as exc:
        # Build a readable pipeline (replace raw vector with placeholder)
        display_pipeline = _make_display_pipeline(pipeline)
        return {
            "query": q,
            "generated_pipeline": {"collection": collection_name, "pipeline": display_pipeline},
            "error": f"Aggregation error: {str(exc)}",
            "llm_latency_ms": llm_ms,
            "embed_latency_ms": embed_ms,
            "usage": usage,
            "vector_search": vector_search_meta or None,
        }
    query_ms = round((time.time() - t1) * 1000)

    display_pipeline = _make_display_pipeline(pipeline)

    return _serialize({
        "query": q,
        "generated_pipeline": {"collection": collection_name, "pipeline": display_pipeline},
        "results": results[:50],
        "result_count": len(results),
        "llm_latency_ms": llm_ms,
        "embed_latency_ms": embed_ms,
        "query_latency_ms": query_ms,
        "usage": usage,
        "vector_search": vector_search_meta or None,
    })


def _make_display_pipeline(pipeline: list) -> list:
    """Replace raw queryVector arrays with a readable placeholder for the UI."""
    out = []
    for stage in pipeline:
        if "$vectorSearch" in stage:
            vs = dict(stage["$vectorSearch"])
            qv = vs.get("queryVector")
            if isinstance(qv, list) and len(qv) > 4:
                vs["queryVector"] = f"<{len(qv)}-dim float vector>"
            out.append({"$vectorSearch": vs})
        else:
            out.append(stage)
    return out


# ──────────────────────────────────────────────────────────
# Document Intelligence: Bedrock Claude multimodal extraction → MongoDB
# ──────────────────────────────────────────────────────────

ALLOWED_UPLOAD_TYPES: dict[str, str] = {
    "application/pdf": "document",
    "image/png": "image",
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/webp": "image",
}

EXTRACTION_PROMPT = (
    "You are a document data extraction specialist for a bank.\n"
    "Analyze this uploaded document and extract ALL text, numbers, and structured data you can find.\n"
    "Return your response in the following JSON format (no markdown fences):\n"
    "{\n"
    '  "document_type": "<type e.g. payslip, bank_statement, invoice, id_card, utility_bill, etc.>",\n'
    '  "extracted_fields": { "<field_name>": "<value>", ... },\n'
    '  "full_text": "<all readable text from the document>",\n'
    '  "confidence": "<high|medium|low>",\n'
    '  "summary": "<1-2 sentence summary of what this document contains>"\n'
    "}\n"
    "Extract every field you can identify (names, dates, amounts, account numbers, addresses, etc.)."
)


@app.post("/api/loan-application/upload")
async def api_loan_upload(
    request: Request,
    file: UploadFile = File(...),
    customer_id: str = Query(...),
) -> dict[str, Any]:
    """
    Accept a document upload (PDF, PNG, JPG), send it to Amazon Bedrock Claude
    for intelligent extraction, store the results in MongoDB.
    """
    content_type = (file.content_type or "").lower()
    ext = (file.filename or "").rsplit(".", 1)[-1].lower() if file.filename else ""
    ext_to_mime = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
    if content_type not in ALLOWED_UPLOAD_TYPES:
        content_type = ext_to_mime.get(ext, content_type)
    if content_type not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Accepted: PDF, PNG, JPG, WEBP")

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

    file_b64 = base64.b64encode(file_bytes).decode("utf-8")
    block_type = ALLOWED_UPLOAD_TYPES[content_type]

    if block_type == "document":
        media_block = {
            "type": "document",
            "source": {"type": "base64", "media_type": content_type, "data": file_b64},
        }
    else:
        media_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": content_type, "data": file_b64},
        }

    messages = [
        {
            "role": "user",
            "content": [
                media_block,
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        }
    ]

    llm_url: str = request.app.state.llm_url
    t0 = time.time()
    try:
        resp = requests.post(llm_url, json={"messages": messages}, timeout=90)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Bedrock extraction failed: {exc}") from exc
    extraction_ms = round((time.time() - t0) * 1000)

    data = resp.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=f"Bedrock error: {data['error']}")

    raw_answer = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    bedrock_model = data.get("model", "")
    usage = data.get("usage", {})

    extracted: dict[str, Any] = {}
    try:
        extracted = json.loads(raw_answer)
    except json.JSONDecodeError:
        cleaned = raw_answer
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].rsplit("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].rsplit("```", 1)[0]
        try:
            extracted = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            extracted = {"full_text": raw_answer, "document_type": "unknown", "confidence": "low"}

    try:
        cust_oid = ObjectId(customer_id)
    except InvalidId:
        cust_oid = None

    record = {
        "customer_id": cust_oid,
        "filename": file.filename,
        "content_type": content_type,
        "file_size_bytes": len(file_bytes),
        "document_type": extracted.get("document_type", "unknown"),
        "extracted_fields": extracted.get("extracted_fields", {}),
        "full_text": extracted.get("full_text", ""),
        "summary": extracted.get("summary", ""),
        "confidence": extracted.get("confidence", ""),
        "extraction_method": "bedrock_claude",
        "bedrock_model": bedrock_model,
        "extraction_ms": extraction_ms,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "uploaded_at": datetime.utcnow(),
        "type": "loan_support_document",
    }

    db = request.app.state.db
    result = db.loan_support_docs.insert_one(record)
    record_id = str(result.inserted_id)

    return _serialize({
        "status": "success",
        "document_id": record_id,
        "filename": file.filename,
        "content_type": content_type,
        "document_type": extracted.get("document_type", "unknown"),
        "extracted_fields": extracted.get("extracted_fields", {}),
        "full_text": extracted.get("full_text", "")[:5000],
        "summary": extracted.get("summary", ""),
        "confidence": extracted.get("confidence", ""),
        "extraction_method": "bedrock_claude",
        "bedrock_model": bedrock_model,
        "extraction_ms": extraction_ms,
        "usage": usage,
        "mongodb_collection": "banking.loan_support_docs",
        "mongodb_document_id": record_id,
    })


@app.post("/api/loan-application/vectorize")
async def api_loan_vectorize(
    request: Request,
    document_id: str = Query(...),
) -> dict[str, Any]:
    """
    Generate a Voyage AI embedding for a previously extracted document
    and store it back in MongoDB — completing the vector-ready pipeline.
    """
    db = request.app.state.db
    voyage_client: voyageai.Client = request.app.state.voyage

    doc_oid = parse_object_id(document_id, "document_id")
    doc = db.loan_support_docs.find_one({"_id": doc_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    text_to_embed = doc.get("summary", "") + "\n" + doc.get("full_text", "")
    fields = doc.get("extracted_fields", {})
    if fields:
        text_to_embed += "\n" + " ".join(f"{k}: {v}" for k, v in fields.items())
    text_to_embed = text_to_embed.strip()
    if not text_to_embed:
        raise HTTPException(status_code=400, detail="No extracted text to vectorize")

    t0 = time.time()
    embedding = get_query_embedding(voyage_client, text_to_embed[:8000])
    embed_ms = round((time.time() - t0) * 1000)

    db.loan_support_docs.update_one(
        {"_id": doc_oid},
        {"$set": {
            EMBEDDING_FIELD: embedding,
            "embedding_model": VOYAGE_EMBED_MODEL,
            "embedding_dimensions": len(embedding),
            "vectorized_at": datetime.utcnow(),
        }},
    )

    return {
        "status": "success",
        "document_id": document_id,
        "embedding_model": VOYAGE_EMBED_MODEL,
        "dimensions": len(embedding),
        "embed_latency_ms": embed_ms,
        "text_length": len(text_to_embed),
        "mongodb_collection": "banking.loan_support_docs",
    }

