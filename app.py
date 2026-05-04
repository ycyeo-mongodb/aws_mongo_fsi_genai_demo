"""
Full FastAPI application for the banking GenAI workshop.
"""

from __future__ import annotations

import json
import os
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

