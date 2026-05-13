"""
LangGraph-powered investigator agent for the banking network_graph.

The agent is a 5-node state machine:
    intent_classifier  (LLM)    Classify the user's question.
    entity_resolver    (logic)  Extract / pivot the target entity from catalog + history.
    pipeline_planner   (logic)  Build the exact MongoDB aggregation to run.
    mongo_executor     (Mongo)  Execute $graphLookup and serialize nodes/edges.
    narrative_writer   (LLM)    Generate a grounded plain-English answer.

Each node appends a structured `step` entry to state.steps so the frontend
can show the full trace (input, output, timing, code, kind).
"""

from __future__ import annotations

import inspect
import json
import re
import time
from operator import add
from typing import Annotated, Any, TypedDict

import requests
from langgraph.graph import END, START, StateGraph
from pymongo.collection import Collection


# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

GRAPH_COL_NAME = "network_graph"
GRAPH_NODE_LIMIT = 600
LLM_TIMEOUT_S = 55
MAX_HISTORY_TURNS = 6

INTENT_CATEGORIES = (
    "explore_entity",       # "tell me about CUST-00007"
    "explain_pattern",      # "why is he flagged?"
    "meta_question",        # "where is this stored?" / "show me the raw document"
    "general_question",     # everything else
)


# ──────────────────────────────────────────────────────────
# Detection pipelines — kept as constants so the UI can SHOW
# the exact aggregation that flagged each risk pattern.
# Mirrors `_detect_risk_alerts` in app.py.
# ──────────────────────────────────────────────────────────

ALERT_DETECTION_PIPELINES: dict[str, dict] = {
    "synthetic_identity_ring": {
        "title": "Shared address/phone/device across 2+ customers",
        "where": "banking.network_graph",
        "pipeline": [
            {"$match": {"type": {"$in": ["Address", "Phone", "Device"]},
                        "metadata.shared_by": {"$gte": 2}}},
            {"$lookup": {
                "from": GRAPH_COL_NAME, "localField": "_id",
                "foreignField": "relationships.target", "as": "linked_back",
            }},
            {"$project": {
                "_id": 1, "type": 1, "label": 1,
                "shared_by": "$metadata.shared_by",
                "customers": {"$map": {
                    "input": {"$filter": {"input": "$linked_back",
                                          "cond": {"$eq": ["$$this.type", "Customer"]}}},
                    "as": "c", "in": "$$c._id"}},
            }},
            {"$sort": {"shared_by": -1}},
        ],
    },
    "shared_contact": {
        "title": "Shared phone — derived from same pipeline as synthetic_identity_ring",
        "where": "banking.network_graph",
        "pipeline": "(same pipeline as synthetic_identity_ring; classification differs by node type)",
    },
    "aml_layering": {
        "title": "Structured wires below the $10k reporting threshold",
        "where": "banking.network_graph",
        "pipeline": [
            {"$match": {"relationships": {"$elemMatch": {"chain_step": {"$exists": True}}}}},
            {"$project": {"_id": 1, "label": 1, "type": 1,
                          "wires": {"$filter": {"input": "$relationships",
                                                "cond": {"$ifNull": ["$$this.chain_step", False]}}}}},
            {"$unwind": "$wires"},
            {"$match": {"wires.type": "WIRE_SENT"}},
            {"$group": {
                "_id": None,
                "accounts": {"$addToSet": "$_id"},
                "targets":  {"$addToSet": "$wires.target"},
                "hops":     {"$sum": 1},
                "total_amount": {"$sum": "$wires.amount_usd"},
                "first_step": {"$min": {"$cond": [
                    {"$eq": ["$wires.chain_step", 1]}, "$_id", None]}},
            }},
        ],
    },
    "ghost_beneficiary": {
        "title": "Customer named as beneficiary on 3+ unrelated policies",
        "where": "banking.network_graph",
        "pipeline": [
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
        ],
    },
    "guarantor_concentration": {
        "title": "Customer guaranteeing 3+ active loans",
        "where": "banking.network_graph",
        "pipeline": [
            {"$match": {"type": "Customer"}},
            {"$addFields": {"guar_loans": {
                "$map": {
                    "input": {"$filter": {"input": "$relationships",
                                          "cond": {"$eq": ["$$this.type", "GUARANTOR_OF"]}}},
                    "as": "r", "in": "$$r.target"}}}},
            {"$match": {"guar_loans.0": {"$exists": True}}},
            {"$addFields": {"guar_count": {"$size": "$guar_loans"}}},
            {"$match": {"guar_count": {"$gte": 3}}},
            {"$sort": {"guar_count": -1}},
        ],
    },
}


# ──────────────────────────────────────────────────────────
# Provenance recipes
# Each recipe explains how a flag is set in production AND runs
# real corroborating queries across multiple collections so the
# audience can see the multi-source derivation, not just the bool.
# ──────────────────────────────────────────────────────────

def _recipe_watchlist(db, entity_id: str, raw_root: dict, alerts: list) -> dict:
    """How `metadata.watchlist` is derived — three corroborating queries."""
    checks: list[dict] = []

    # ── Check 1: KYC documents with risk flags ─────────────
    t = time.time()
    kyc_pipeline = [
        {"$match": {"customer_id": entity_id, "risk_flags": {"$ne": []}}},
        {"$project": {"_id": 0, "id": 1, "document_type": 1,
                      "verification_status": 1, "risk_flags": 1, "expiry_date": 1}},
    ]
    kyc_results = list(db.kyc_documents.aggregate(kyc_pipeline))
    checks.append({
        "step": 1,
        "title": "Adverse KYC findings (banking.kyc_documents)",
        "operation": "aggregate",
        "collection": "banking.kyc_documents",
        "pipeline": kyc_pipeline,
        "result_count": len(kyc_results),
        "results": kyc_results[:5],
        "execution_ms": round((time.time() - t) * 1000, 1),
        "signal_matched": len(kyc_results) > 0,
        "why_this_matters": "KYC docs flagged with risk indicators (name_mismatch, expired ID, suspicious photo) are a direct input to the watchlist screening batch.",
    })

    # ── Check 2: Customer risk profile from canonical customers collection ─
    t = time.time()
    cust_query = {"id": entity_id}
    cust_proj = {"_id": 0, "id": 1, "name": 1, "risk_level": 1,
                 "credit_score": 1, "payment_history": 1, "employment_type": 1}
    cust_result = db.customers.find_one(cust_query, cust_proj) or {}
    risky = cust_result.get("risk_level") in ("high", "medium") or \
            cust_result.get("payment_history") in ("poor", "late")
    checks.append({
        "step": 2,
        "title": "Canonical risk attributes (banking.customers)",
        "operation": "findOne",
        "collection": "banking.customers",
        "query": cust_query,
        "projection": cust_proj,
        "result_count": 1 if cust_result else 0,
        "results": [cust_result] if cust_result else [],
        "execution_ms": round((time.time() - t) * 1000, 1),
        "signal_matched": bool(risky),
        "why_this_matters": "The customer master record holds the risk_level, credit_score and payment_history attributes that compliance ingests nightly. risk_level in (high|medium) or poor payment history feeds the watchlist consolidator.",
    })

    # ── Check 3: Count of pre-detected risk alerts focused on this entity ──
    t = time.time()
    matching_alerts = [
        {"kind": a.get("kind"), "severity": a.get("severity"), "title": a.get("title")}
        for a in (alerts or [])
        if a.get("focus_node") == entity_id or entity_id in (a.get("highlight_nodes") or [])
    ]
    checks.append({
        "step": 3,
        "title": "Active risk alerts naming this entity (in-memory derivation)",
        "operation": "aggregate (already computed at request start)",
        "collection": "banking.network_graph",
        "pipeline": "(see ALERT_DETECTION_PIPELINES — these 4 aggregations run once per request)",
        "result_count": len(matching_alerts),
        "results": matching_alerts[:5],
        "execution_ms": round((time.time() - t) * 1000, 1),
        "signal_matched": len(matching_alerts) > 0,
        "why_this_matters": "Real-time pattern detection across the graph: any entity that appears as the focus of an active alert (ghost beneficiary, guarantor concentration, AML chain, synthetic-identity ring) is auto-promoted onto the watchlist.",
    })

    matched = sum(1 for c in checks if c["signal_matched"])
    return {
        "production_data_flow": (
            "In production this boolean is set by a nightly compliance batch that "
            "aggregates from five upstream sources: (1) OFAC SDN / EU / UN sanctions "
            "list match against name + DOB; (2) Adverse media monitoring (World-Check, "
            "Dow Jones, LexisNexis); (3) PEP (Politically Exposed Person) lookups; "
            "(4) Outcomes of past Suspicious Activity Reports / internal investigations; "
            "(5) Behavioural pattern alerts. The boolean on network_graph.metadata is the "
            "materialized OR of these inputs — denormalised so screening dashboards stay "
            "fast at scale."
        ),
        "demo_note": (
            "External screening systems are not modelled in this demo, but the flag is "
            "auditable against the underlying behavioural data through these three live "
            "cross-collection queries:"
        ),
        "checks": checks,
        "verification_summary": (
            f"{matched} of {len(checks)} corroborating signals matched against live data."
        ),
    }


def _recipe_aml_flag(db, entity_id: str, raw_root: dict, alerts: list) -> dict:
    """How `metadata.aml_flag` is derived — three corroborating signals."""
    checks: list[dict] = []
    rel_list = (raw_root or {}).get("relationships") or []

    # ── Check 1: Chain-step wire transfers (structuring signature) ─────────
    t = time.time()
    chain_pipeline = [
        {"$match": {"_id": entity_id}},
        {"$project": {
            "chain_wires": {"$filter": {
                "input": "$relationships",
                "cond": {"$ifNull": ["$$this.chain_step", False]},
            }},
        }},
        {"$unwind": {"path": "$chain_wires", "preserveNullAndEmptyArrays": False}},
        {"$project": {
            "_id": 0,
            "step": "$chain_wires.chain_step",
            "type": "$chain_wires.type",
            "target": "$chain_wires.target",
            "amount_usd": "$chain_wires.amount_usd",
        }},
        {"$sort": {"step": 1}},
    ]
    chain_results = list(db.network_graph.aggregate(chain_pipeline))
    # Also check accounts the customer owns
    if not chain_results:
        owned = [r["target"] for r in rel_list if r.get("type") == "OWNS_ACCOUNT"]
        if owned:
            chain_pipeline_acct = [
                {"$match": {"_id": {"$in": owned}}},
                {"$project": {
                    "chain_wires": {"$filter": {
                        "input": "$relationships",
                        "cond": {"$ifNull": ["$$this.chain_step", False]},
                    }},
                }},
                {"$unwind": "$chain_wires"},
                {"$project": {"_id": 0, "account_id": "$_id",
                              "step": "$chain_wires.chain_step",
                              "amount_usd": "$chain_wires.amount_usd"}},
                {"$sort": {"step": 1}},
            ]
            chain_results = list(db.network_graph.aggregate(chain_pipeline_acct))
            chain_pipeline = chain_pipeline_acct
    checks.append({
        "step": 1,
        "title": "Structured wire chain (banking.network_graph)",
        "operation": "aggregate",
        "collection": "banking.network_graph",
        "pipeline": chain_pipeline,
        "result_count": len(chain_results),
        "results": chain_results[:8],
        "execution_ms": round((time.time() - t) * 1000, 1),
        "signal_matched": len(chain_results) > 0,
        "why_this_matters": "Wires marked with `chain_step` are the structuring signature: a sequence of transfers each just below the $10k reporting threshold designed to evade Currency Transaction Reports. This is one of the strongest individual AML signals.",
    })

    # ── Check 2: Guarantor-of count from relationships array ───────────────
    t = time.time()
    guar_pipeline = [
        {"$match": {"_id": entity_id}},
        {"$project": {
            "_id": 0,
            "guar_loans": {"$map": {
                "input": {"$filter": {"input": "$relationships",
                                      "cond": {"$eq": ["$$this.type", "GUARANTOR_OF"]}}},
                "as": "r", "in": "$$r.target",
            }},
        }},
        {"$addFields": {"guar_count": {"$size": "$guar_loans"}}},
    ]
    guar_results = list(db.network_graph.aggregate(guar_pipeline))
    guar_count = guar_results[0]["guar_count"] if guar_results else 0
    checks.append({
        "step": 2,
        "title": "Guarantor-of obligations (banking.network_graph)",
        "operation": "aggregate",
        "collection": "banking.network_graph",
        "pipeline": guar_pipeline,
        "result_count": len(guar_results),
        "results": guar_results,
        "execution_ms": round((time.time() - t) * 1000, 1),
        "signal_matched": guar_count >= 3,
        "why_this_matters": "Customers guaranteeing 3+ active loans signal concentration risk. Combined with adverse KYC findings this typically triggers an Enhanced Due Diligence review.",
    })

    # ── Check 3: Insurance policies where this entity is beneficiary ───────
    t = time.time()
    benef_pipeline = [
        {"$match": {"holder_id": {"$ne": entity_id}, "beneficiary_ids": entity_id}},
        {"$project": {"_id": 0, "id": 1, "policy_type": 1, "coverage_usd": 1,
                      "holder_id": 1, "underwriter": 1, "status": 1}},
    ]
    benef_results = list(db.insurance_policies.aggregate(benef_pipeline))
    checks.append({
        "step": 3,
        "title": "Beneficiary on unrelated policies (banking.insurance_policies)",
        "operation": "aggregate",
        "collection": "banking.insurance_policies",
        "pipeline": benef_pipeline,
        "result_count": len(benef_results),
        "results": benef_results[:5],
        "execution_ms": round((time.time() - t) * 1000, 1),
        "signal_matched": len(benef_results) >= 3,
        "why_this_matters": "Being beneficiary on 3+ unrelated policies (different holders, no family link) is the ghost-beneficiary fraud signature. Combined with watchlist status it almost always upgrades the AML flag.",
    })

    matched = sum(1 for c in checks if c["signal_matched"])
    return {
        "production_data_flow": (
            "In production this flag is set by the Transaction Monitoring System "
            "(typically Actimize, SAS AML, or Oracle FCC) which runs rule-based "
            "scenarios over: (1) Transaction velocity and size (structuring detection); "
            "(2) Geographic risk scoring against FATF high-risk jurisdictions; "
            "(3) Counterparty risk (shared identifiers, common addresses); "
            "(4) Outcome of previously filed Suspicious Activity Reports. "
            "The flag is set when 2+ scenarios fire above their respective thresholds, "
            "and is reviewed by an AML analyst before promotion to a SAR."
        ),
        "demo_note": (
            "We re-derive the same conclusion in real time by combining three signals "
            "from three different MongoDB collections:"
        ),
        "checks": checks,
        "verification_summary": (
            f"{matched} of {len(checks)} corroborating signals matched against live data."
        ),
    }


def _recipe_shared_by(db, entity_id: str, raw_root: dict, alerts: list) -> dict:
    """How `metadata.shared_by` is derived — for Address/Phone/Device nodes."""
    checks: list[dict] = []

    # ── Check 1: Live count of customers linking back to this node ────────
    t = time.time()
    pipeline = [
        {"$match": {"_id": entity_id}},
        {"$lookup": {
            "from": GRAPH_COL_NAME,
            "localField": "_id",
            "foreignField": "relationships.target",
            "as": "linked_back",
        }},
        {"$project": {
            "_id": 1, "type": 1, "label": 1,
            "cached_shared_by": "$metadata.shared_by",
            "live_linked_customers": {"$map": {
                "input": {"$filter": {"input": "$linked_back",
                                      "cond": {"$eq": ["$$this.type", "Customer"]}}},
                "as": "c", "in": {"id": "$$c._id", "label": "$$c.label",
                                  "watchlist": "$$c.metadata.watchlist",
                                  "aml_flag": "$$c.metadata.aml_flag"},
            }},
        }},
        {"$addFields": {"live_count": {"$size": "$live_linked_customers"}}},
    ]
    results = list(db.network_graph.aggregate(pipeline))
    live_count = results[0]["live_count"] if results else 0
    checks.append({
        "step": 1,
        "title": "Live count of customers sharing this attribute",
        "operation": "aggregate",
        "collection": "banking.network_graph",
        "pipeline": pipeline,
        "result_count": len(results),
        "results": results,
        "execution_ms": round((time.time() - t) * 1000, 1),
        "signal_matched": live_count >= 2,
        "why_this_matters": "The cached shared_by counter is rebuilt nightly. This aggregation re-derives the same value from live edges, proving the cache is correct.",
    })

    matched = sum(1 for c in checks if c["signal_matched"])
    return {
        "production_data_flow": (
            "Each Address / Phone / Device entity stores a `shared_by` counter pointing "
            "to the number of distinct Customer documents currently linking to it. The "
            "counter is built incrementally during the onboarding pipeline (`setup_graph.py` "
            "in this repo, an event-driven ingestion in production) and validated by a "
            "nightly reconciliation aggregation. A value of >= 2 trips the synthetic-identity "
            "alert because two unrelated customers should almost never share an exact "
            "address/device fingerprint."
        ),
        "demo_note": (
            "The cached field can be re-derived at any time with a single $lookup:"
        ),
        "checks": checks,
        "verification_summary": (
            f"{matched} of {len(checks)} corroborating signals matched against live data."
        ),
    }


PROVENANCE_RECIPES = {
    "watchlist":  _recipe_watchlist,
    "aml_flag":   _recipe_aml_flag,
    "shared_by":  _recipe_shared_by,
}

# Regex patterns for common entity-ID shapes in this demo.
ENTITY_ID_PATTERNS = [
    re.compile(r"\b(CUST-\d{3,6})\b", re.I),
    re.compile(r"\b(LOAN-\d{3,6})\b", re.I),
    re.compile(r"\b(POL-\d{3,6})\b", re.I),
    re.compile(r"\b(ACC-\d{3,6})\b", re.I),
    re.compile(r"\b(ADDR-\d{3,6})\b", re.I),
    re.compile(r"\b(PHN-\d{3,6})\b", re.I),
    re.compile(r"\b(DEV-[A-Z0-9-]+)\b", re.I),
    re.compile(r"\b(EMP-\d{3,6})\b", re.I),
    re.compile(r"\b(CP-\d{3,6})\b", re.I),
]


# ──────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────

class InvestigatorState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────
    query: str
    history: list[dict]
    catalog: str                 # prompt-ready listing of every node
    alerts: list[dict]           # precomputed risk alerts
    catalog_ids: set             # quick membership check for ID validation
    show_all_relationships: bool # UI toggle: bump depth + skip edge trim

    # ── Per-node outputs ──────────────────────────────────
    intent: dict                 # {category, entity_hints, depth, reasoning}
    entities: list[str]          # resolved entity IDs to focus on
    plan: dict                   # {action, depth, pipeline, show_all}
    pipeline: list[dict]         # the actual aggregation stages
    graph_nodes: list[dict]      # serialized for frontend
    graph_edges: list[dict]
    raw_root_document: dict      # the BSON document we read for the root entity
    highlight_nodes: list[str]
    evidence: dict               # structured provenance for every claim
    narrative: str

    # ── Trace ─────────────────────────────────────────────
    steps: Annotated[list[dict], add]


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _llm_call(llm_url: str, messages: list[dict], *, max_tokens: int = 1200) -> str:
    """Call the API-Gateway-fronted LLM and return raw assistant content."""
    resp = requests.post(
        llm_url,
        json={"messages": messages, "max_tokens": max_tokens},
        timeout=LLM_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()


def _parse_json_loose(raw: str) -> dict | None:
    """Tolerate ```json fences around model output."""
    if not raw:
        return None
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.MULTILINE).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def _serialize_node(doc: dict, hops: int | None = None, is_root: bool = False) -> dict:
    return {
        "id": doc["_id"],
        "label": doc.get("label") or doc["_id"],
        "type": doc.get("type", "Unknown"),
        "metadata": doc.get("metadata", {}),
        "doc_count": len(doc.get("relationships", [])),
        "hops": hops,
        "is_root": is_root,
    }


def _flatten_edges(doc: dict) -> list[dict]:
    out = []
    for rel in doc.get("relationships", []) or []:
        edge = {"from": doc["_id"], "to": rel.get("target"), "label": rel.get("type", "")}
        for k in (
            "source", "amount_usd", "transaction_count", "total_amount_usd",
            "date", "chain_step", "loan_id", "description",
        ):
            if k in rel:
                edge[k] = rel[k]
        out.append(edge)
    return out


def _last_focused_entity(history: list[dict]) -> str | None:
    """Look back through assistant messages for the last entity ID that was investigated."""
    if not history:
        return None
    for h in reversed(history):
        if h.get("role") != "assistant":
            continue
        meta = h.get("meta") or {}
        if isinstance(meta.get("entities"), list) and meta["entities"]:
            return meta["entities"][0]
        # fall back: scan the content for an ID pattern
        content = str(h.get("content") or "")
        for pat in ENTITY_ID_PATTERNS:
            m = pat.search(content)
            if m:
                return m.group(1).upper()
    return None


def _step(node: str, kind: str, title: str, *, fn=None, **extras) -> dict:
    """Build a single trace entry. `fn` lets us embed real source code."""
    entry: dict[str, Any] = {"node": node, "kind": kind, "title": title}
    if fn is not None:
        try:
            entry["code"] = inspect.getsource(fn)
        except (OSError, TypeError):
            entry["code"] = None
    entry.update(extras)
    return entry


# ──────────────────────────────────────────────────────────
# LangGraph Nodes
# ──────────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are the FIRST stage of a 6-stage LangGraph investigator agent for a bank's network_graph.
Your single job: classify the user's question and surface any entity IDs they mention or imply.

You will see:
- The user question (latest turn)
- The recent conversation history (so you can resolve "him", "her", "that customer")
- A compact CATALOG of every node in the graph (ID | TYPE | LABEL [meta])
- A list of pre-detected RISK ALERTS with focus_node IDs

Return STRICT JSON (no markdown fences) of this exact shape:
{
  "category": "explore_entity" | "explain_pattern" | "meta_question" | "general_question",
  "entity_hints": ["<exact node _id>", ...],     // 0..N IDs from the catalog
  "depth": 1 | 2 | 3,                             // graph traversal depth, 2 is default
  "reasoning": "<one sentence>"
}

CATEGORY GUIDE — pick exactly one:

• "explore_entity"   — user asks about a specific customer / account / policy by ID,
                       name, or pronoun referring to a prior turn.
                       Examples: "tell me about CUST-00007", "show me Vann Vanna's network".

• "explain_pattern"  — user asks WHY something is flagged or HOW a known risk pattern
                       works. Still surface the relevant focus_node in entity_hints.
                       Examples: "why is he flagged?", "what's the riskiest pattern?",
                                 "show me the AML chain".

• "meta_question"    — user is asking about DATA PROVENANCE, storage, or how the
                       investigator knows what it knows. The user wants to see the
                       raw source data, not a graph traversal.
                       Examples: "where is this data stored?",
                                 "how do you know about the AML flag?",
                                 "show me the raw document for CUST-00007",
                                 "which collection holds the watchlist field?",
                                 "is this real or are you making it up?",
                                 "can you cite the source?".
                       If a specific entity is implied (from history or named), put it
                       in entity_hints so the agent can fetch its raw document.

• "general_question" — meta questions like "what can I ask?" or capability queries
                       with no specific data lookup needed.

OTHER RULES:
- Resolve vague references using the catalog. "the ghost beneficiary" → find a
  Customer with ghost_beneficiary alert and return its ID.
- If the user follows up ("why is he flagged?") and the last assistant turn was
  about CUST-00007, keep CUST-00007 in entity_hints.
- depth=1 for "who owns X" / single-hop; depth=2 for "tell me about X" / fraud-ring;
  depth=3 only for transitive multi-hop AML traces. For meta_question, depth=1 (we
  only fetch the single root document).
- Use EXACT IDs from the catalog left column. Do NOT invent IDs."""


NARRATIVE_SYSTEM_PROMPT = """You are the FINAL stage of a 6-stage LangGraph investigator agent.
The earlier stages have already executed the MongoDB query AND built an EVIDENCE PACKAGE
that records exactly where every fact came from. Your job: write a concise, factual answer
for a compliance officer that is GROUNDED in the evidence package — never invent facts that
are not present in the evidence.

You will receive a JSON payload with:
- user_question, intent (category + reasoning)
- subgraph_summary (counts of node types, edge types, flagged nodes)
- evidence: {
    root_document_excerpt: { _id, type, label, metadata } from banking.network_graph,
    field_facts: [ {claim, source_path, value}, ... ],   // direct field reads
    alert_provenance: [ {kind, severity, title, summary,
                         detection_pipeline_id} ]        // aggregations that fired
  }

RULES — read carefully:

1.  Every concrete claim you make (watchlist, AML flag, "guarantor on N loans",
    "beneficiary on N policies", credit score, shared address/device) MUST be
    present in evidence.field_facts or evidence.alert_provenance. If a fact is
    NOT in evidence, do NOT state it.

2.  For meta_question intent, EXPLAIN where the data lives. Reference the
    collection name (banking.network_graph), the document _id, and the specific
    field paths (e.g. metadata.watchlist). Say "this came from a MongoDB
    aggregation pipeline (visible in the Evidence panel below)" — do NOT
    reproduce the pipeline JSON in your prose.

3.  For explore_entity / explain_pattern, write 2-5 sentences. Plain prose.
    No markdown headings, no bullets. Reference labels (e.g. "Vann Vanna
    (CUST-00007)") rather than only IDs.

4.  For general_question, answer in 2-3 sentences. If there is no evidence
    package (because no Mongo query ran), say so explicitly: "I did not query
    the database for this question — here is general guidance." This signals
    to the audience that the answer is general knowledge, not grounded in
    their data.

5.  Tone: confident, factual, no hedging. The compliance officer needs to act
    on this, not parse it."""


def node_evidence_builder(state: InvestigatorState, db) -> dict:
    """
    Pure logic node. Inspects the data the executor returned and assembles a
    structured `evidence` dict that:
      • lists every field-level fact (watchlist, aml_flag, credit_score, …) with
        its source path in banking.network_graph AND a provenance recipe
        explaining how the flag is set in production plus live corroborating
        queries against OTHER collections,
      • attaches the actual aggregation pipelines that triggered any alert the
        root entity is part of,
      • includes a trimmed copy of the raw root document.
    The narrative_writer is required to ground its answer in this evidence.
    """
    t0 = time.time()

    raw_root: dict = state.get("raw_root_document") or {}
    nodes = state.get("graph_nodes") or []
    edges = state.get("graph_edges") or []
    alerts = state.get("alerts") or []
    plan = state.get("plan") or {}
    target = plan.get("target_entity")

    evidence: dict[str, Any] = {
        "source_collection": f"banking.{GRAPH_COL_NAME}",
        "mongodb_operation": plan.get("action") or "none",
        "root_document_excerpt": None,
        "field_facts": [],
        "neighbour_summary": {},
        "alert_provenance": [],
    }

    # ── 1. Root document excerpt (trim relationships to ≤15 to keep payload sane)
    if raw_root:
        excerpt = {
            "_id": raw_root.get("_id"),
            "type": raw_root.get("type"),
            "label": raw_root.get("label"),
            "metadata": raw_root.get("metadata") or {},
            "normalized_label": raw_root.get("normalized_label"),
            "relationships_count": len(raw_root.get("relationships") or []),
            "relationships_sample": (raw_root.get("relationships") or [])[:15],
        }
        evidence["root_document_excerpt"] = excerpt

        # ── 2. Field-level facts — direct reads from metadata
        rid = raw_root["_id"]
        m: dict = raw_root.get("metadata") or {}
        col = f"banking.{GRAPH_COL_NAME}"

        def fact(claim: str, value: Any, field: str, kind: str = "field_read") -> dict:
            return {
                "claim": claim,
                "source_path": f"{col}.findOne({{_id: '{rid}'}}).{field}",
                "value": value,
                "kind": kind,
            }

        # Helper that adds a fact AND optionally attaches a provenance recipe
        # showing how that field is set in production + live corroborating
        # queries across other collections.
        def add_fact_with_provenance(field_name: str, claim: str, value: Any,
                                      field_path: str, kind: str = "field_read") -> None:
            f = {
                "claim": claim,
                "source_path": f"{col}.findOne({{_id: '{rid}'}}).{field_path}",
                "value": value,
                "kind": kind,
                "field": field_name,
            }
            recipe = PROVENANCE_RECIPES.get(field_name)
            if recipe and db is not None:
                try:
                    f["provenance"] = recipe(db, rid, raw_root, alerts)
                except Exception as e:
                    f["provenance"] = {
                        "production_data_flow": "(unavailable in this run)",
                        "demo_note": f"Provenance computation failed: {e}",
                        "checks": [],
                        "verification_summary": "0 of 0 corroborating signals matched.",
                    }
            evidence["field_facts"].append(f)

        if m.get("watchlist"):
            add_fact_with_provenance(
                "watchlist", "On the bank's watchlist",
                True, "metadata.watchlist",
            )
        if m.get("aml_flag"):
            add_fact_with_provenance(
                "aml_flag", "AML compliance flag raised",
                True, "metadata.aml_flag",
            )
        if (m.get("shared_by") or 0) >= 2:
            add_fact_with_provenance(
                "shared_by", f"Shared by {m['shared_by']} customers",
                int(m["shared_by"]), "metadata.shared_by",
            )
        if m.get("credit_score") is not None:
            evidence["field_facts"].append(fact(
                f"Credit score: {m['credit_score']}",
                m["credit_score"], "metadata.credit_score",
            ))
        if m.get("monthly_income_usd") is not None:
            evidence["field_facts"].append(fact(
                f"Monthly income: ${m['monthly_income_usd']:,}",
                m["monthly_income_usd"], "metadata.monthly_income_usd",
            ))
        if m.get("balance_usd") is not None:
            evidence["field_facts"].append(fact(
                f"Account balance: ${m['balance_usd']:,}",
                m["balance_usd"], "metadata.balance_usd",
            ))
        if m.get("amount_usd") is not None:
            evidence["field_facts"].append(fact(
                f"Loan amount: ${m['amount_usd']:,}",
                m["amount_usd"], "metadata.amount_usd",
            ))
        if m.get("coverage_usd") is not None:
            evidence["field_facts"].append(fact(
                f"Policy coverage: ${m['coverage_usd']:,}",
                m["coverage_usd"], "metadata.coverage_usd",
            ))
        if m.get("policy_type"):
            evidence["field_facts"].append(fact(
                f"Policy type: {m['policy_type']}",
                m["policy_type"], "metadata.policy_type",
            ))
        if m.get("purpose"):
            evidence["field_facts"].append(fact(
                f"Loan purpose: {m['purpose']}",
                m["purpose"], "metadata.purpose",
            ))

        # Relationship-derived counts (computed on the spot, ground truth):
        rel_list = raw_root.get("relationships") or []
        guar_count = sum(1 for r in rel_list if r.get("type") == "GUARANTOR_OF")
        if guar_count:
            evidence["field_facts"].append({
                "claim": f"Guarantor on {guar_count} loan(s)",
                "source_path": f"{col}.findOne({{_id: '{rid}'}}).relationships[type=='GUARANTOR_OF'].length",
                "value": guar_count,
                "kind": "derived_count",
            })
        owns_count = sum(1 for r in rel_list if r.get("type") == "OWNS_ACCOUNT")
        if owns_count:
            evidence["field_facts"].append({
                "claim": f"Owns {owns_count} account(s)",
                "source_path": f"{col}.findOne({{_id: '{rid}'}}).relationships[type=='OWNS_ACCOUNT'].length",
                "value": owns_count,
                "kind": "derived_count",
            })

    # ── 3. Neighbour summary by type (for explore_entity)
    if nodes:
        by_type: dict[str, int] = {}
        flagged: list[dict] = []
        for n in nodes:
            by_type[n.get("type", "?")] = by_type.get(n.get("type", "?"), 0) + 1
            md = n.get("metadata") or {}
            if md.get("watchlist") or md.get("aml_flag"):
                flagged.append({"id": n["id"], "label": n["label"], "type": n["type"]})
        evidence["neighbour_summary"] = {
            "by_type": by_type,
            "flagged_neighbours": flagged[:10],
            "edge_count": len(edges),
        }

    # ── 4. Alert provenance — pull the actual detection pipelines
    if target:
        for a in alerts:
            in_focus = (a.get("focus_node") == target)
            in_highlights = target in (a.get("highlight_nodes") or [])
            if in_focus or in_highlights:
                kind = a.get("kind", "")
                det = ALERT_DETECTION_PIPELINES.get(kind, {})
                evidence["alert_provenance"].append({
                    "kind": kind,
                    "severity": a.get("severity"),
                    "title": a.get("title"),
                    "summary": a.get("summary"),
                    "focus_node": a.get("focus_node"),
                    "detection_collection": det.get("where", f"banking.{GRAPH_COL_NAME}"),
                    "detection_pipeline": det.get("pipeline"),
                    "detection_note": det.get("title"),
                })

    elapsed = round((time.time() - t0) * 1000, 1)
    return {
        "evidence": evidence,
        "steps": [_step(
            "evidence_builder", "logic",
            "Build a citable evidence package from the executor's output",
            fn=node_evidence_builder,
            input={
                "has_raw_root": bool(raw_root),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "alerts_total": len(alerts),
                "target": target,
            },
            output={
                "source_collection": evidence["source_collection"],
                "field_facts_count": len(evidence["field_facts"]),
                "alert_provenance_count": len(evidence["alert_provenance"]),
                "neighbour_types": list(evidence.get("neighbour_summary", {}).get("by_type") or {}),
            },
            ms=elapsed,
        )],
    }


def node_intent_classifier(state: InvestigatorState, llm_url: str) -> dict:
    """LLM call: classify the question and surface entity hints."""
    t0 = time.time()

    # Build history block (last 6 turns)
    hist_block = ""
    for h in (state.get("history") or [])[-MAX_HISTORY_TURNS:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            hist_block += f"{h['role'].upper()}: {str(h['content'])[:600]}\n"
    hist_block = hist_block.strip() or "(no prior turns)"

    alerts = state.get("alerts") or []
    alert_lines = []
    for a in alerts:
        alert_lines.append(
            f"- [{a.get('severity','?').upper()}] {a.get('kind','?')}: "
            f"focus_node={a.get('focus_node','?')} — {a.get('title','')}"
        )
    alerts_block = "\n".join(alert_lines) if alert_lines else "(none)"

    user_block = (
        f"USER QUESTION:\n{state['query']}\n\n"
        f"HISTORY:\n{hist_block}\n\n"
        f"PRE-DETECTED ALERTS:\n{alerts_block}\n\n"
        f"CATALOG:\n{state.get('catalog','')}"
    )

    messages = [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]

    raw = _llm_call(llm_url, messages, max_tokens=400)
    parsed = _parse_json_loose(raw) or {
        "category": "general_question",
        "entity_hints": [],
        "depth": 2,
        "reasoning": "LLM response could not be parsed; defaulting to general_question.",
    }

    # Sanitize
    cat = parsed.get("category") if parsed.get("category") in INTENT_CATEGORIES else "general_question"
    hints = parsed.get("entity_hints") or []
    if not isinstance(hints, list):
        hints = []
    hints = [str(x).strip() for x in hints if x]
    depth = parsed.get("depth", 2)
    try:
        depth = max(1, min(3, int(depth)))
    except (TypeError, ValueError):
        depth = 2

    intent = {
        "category": cat,
        "entity_hints": hints,
        "depth": depth,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }

    elapsed = round((time.time() - t0) * 1000, 1)
    return {
        "intent": intent,
        "steps": [_step(
            "intent_classifier", "llm", "Classify intent & extract entity hints",
            fn=node_intent_classifier,
            input={"query": state["query"], "history_turns": len(state.get("history") or [])},
            output=intent,
            ms=elapsed,
        )],
    }


def node_entity_resolver(state: InvestigatorState) -> dict:
    """Pure logic: combine LLM hints + regex patterns + history pivot + catalog membership."""
    t0 = time.time()
    intent = state.get("intent") or {}
    query = state.get("query", "")
    catalog_ids: set = state.get("catalog_ids") or set()

    resolved: list[str] = []
    seen: set[str] = set()

    # (1) LLM-supplied hints (validated against catalog)
    for h in intent.get("entity_hints") or []:
        cand = h.upper().strip()
        if cand and cand in catalog_ids and cand not in seen:
            resolved.append(cand)
            seen.add(cand)

    # (2) Regex pull from the raw query (in case the LLM dropped them)
    if not resolved:
        for pat in ENTITY_ID_PATTERNS:
            for m in pat.finditer(query):
                cand = m.group(1).upper()
                if cand in catalog_ids and cand not in seen:
                    resolved.append(cand)
                    seen.add(cand)

    # (3) Pronoun / follow-up pivot — pull last focused entity from history.
    # Applies to entity-style questions AND meta-questions ("show me the raw
    # document" usually means: the doc for the entity we were just discussing).
    if not resolved and intent.get("category") in ("explore_entity", "explain_pattern", "meta_question"):
        prior = _last_focused_entity(state.get("history") or [])
        if prior and prior in catalog_ids:
            resolved.append(prior)

    # (4) For explain_pattern with no entity, fall back to highest-severity alert focus_node
    if not resolved and intent.get("category") == "explain_pattern":
        for a in state.get("alerts") or []:
            fn = a.get("focus_node")
            if fn and fn in catalog_ids:
                resolved.append(fn)
                break

    method = (
        "llm_hint" if intent.get("entity_hints") and resolved == [intent["entity_hints"][0].upper()]
        else "regex" if any(p.search(query) for p in ENTITY_ID_PATTERNS)
        else "history_pivot" if resolved
        else "none"
    )

    elapsed = round((time.time() - t0) * 1000, 1)
    return {
        "entities": resolved,
        "steps": [_step(
            "entity_resolver", "logic", "Resolve target entities (regex + catalog + history)",
            fn=node_entity_resolver,
            input={
                "llm_hints": intent.get("entity_hints"),
                "query": query,
                "history_turns": len(state.get("history") or []),
            },
            output={"entities": resolved, "method": method},
            ms=elapsed,
        )],
    }


def node_pipeline_planner(state: InvestigatorState) -> dict:
    """Pure logic: decide what aggregation to run based on intent + entities."""
    t0 = time.time()
    intent = state.get("intent") or {}
    entities = state.get("entities") or []
    show_all = bool(state.get("show_all_relationships", False))
    category = intent.get("category", "general_question")
    depth = int(intent.get("depth", 2))
    # "Show all relationships" bumps traversal one extra hop so the user
    # sees indirect ties (a counterparty's counterparty, the shared device's
    # other owners' loans, etc.). Cap at 3 to keep payload sane.
    if show_all:
        depth = max(depth, 3)

    plan: dict[str, Any] = {
        "action": "none",
        "depth": depth,
        "target_entity": None,
        "show_all": show_all,
    }
    pipeline: list[dict] = []

    if entities and category in ("explore_entity", "explain_pattern"):
        # Traversal — full subgraph
        target = entities[0]
        plan["action"] = "graph_lookup"
        plan["target_entity"] = target
        pipeline = [
            {"$match": {"_id": target}},
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
        ]
    elif entities and category == "meta_question":
        # Provenance — just pull the single raw document. No traversal.
        # The audience needs to SEE the BSON; the agent should not narrate from
        # imagination when a literal find_one() answers the question definitively.
        target = entities[0]
        plan["action"] = "find_one"
        plan["target_entity"] = target
        plan["depth"] = 1
        pipeline = [{"$match": {"_id": target}}, {"$limit": 1}]

    elapsed = round((time.time() - t0) * 1000, 1)
    return {
        "plan": plan,
        "pipeline": pipeline,
        "steps": [_step(
            "pipeline_planner", "logic", "Decide the MongoDB operation",
            fn=node_pipeline_planner,
            input={"category": category, "entities": entities,
                   "depth": depth, "show_all_relationships": show_all},
            output={"plan": plan, "pipeline": pipeline},
            ms=elapsed,
        )],
    }


def node_mongo_executor(state: InvestigatorState, graph: Collection) -> dict:
    """Run the planned aggregation and serialize the result into nodes/edges."""
    t0 = time.time()
    pipeline = state.get("pipeline") or []
    plan = state.get("plan") or {}
    action = plan.get("action")
    show_all = bool(plan.get("show_all"))

    nodes_out: list[dict] = []
    edges_out: list[dict] = []
    highlight: list[str] = []
    raw_root: dict = {}
    result_count = 0
    rows_inspected = 0
    dangling_edges = 0

    if pipeline and action == "graph_lookup":
        result = list(graph.aggregate(pipeline))
        result_count = len(result)
        if result:
            root = result[0]
            # Keep the raw BSON (minus the heavy `connected` array) so the
            # Evidence panel can show the auditor exactly what was read.
            raw_root = {k: v for k, v in root.items() if k != "connected"}
            connected = root.get("connected", []) or []
            rows_inspected = 1 + len(connected)

            nodes_out = [_serialize_node(root, hops=0, is_root=True)]
            seen = {root["_id"]}
            for c in connected:
                if c["_id"] in seen:
                    continue
                nodes_out.append(_serialize_node(c, hops=int(c.get("hops", 0)) + 1))
                seen.add(c["_id"])
                if len(nodes_out) >= GRAPH_NODE_LIMIT:
                    break

            # When "show all relationships" is ON we also surface edges whose
            # target lies outside the rendered set and synthesize a stub node
            # for that target so the edge has somewhere to land visually.
            stub_added: set[str] = set()
            for d in [root] + connected:
                if d["_id"] not in seen:
                    continue
                for e in _flatten_edges(d):
                    if e["to"] in seen:
                        edges_out.append(e)
                    elif show_all and e.get("to"):
                        edges_out.append(e)
                        dangling_edges += 1
                        if e["to"] not in stub_added and len(nodes_out) < GRAPH_NODE_LIMIT:
                            stub_added.add(e["to"])
                            nodes_out.append({
                                "id": e["to"],
                                "label": e["to"],
                                "type": "Unknown",
                                "metadata": {},
                                "doc_count": 0,
                                "hops": None,
                                "is_root": False,
                            })

            highlight = [n["id"] for n in nodes_out if (n.get("metadata") or {}).get("watchlist")
                         or (n.get("metadata") or {}).get("aml_flag")]

    elif pipeline and action == "find_one":
        # Meta-question branch: fetch ONLY the root document. No traversal.
        result = list(graph.aggregate(pipeline))
        result_count = len(result)
        rows_inspected = result_count
        if result:
            root = result[0]
            raw_root = dict(root)
            # Still surface the root as a single graph node so the UI can show
            # it visually — but no edges, no neighbours.
            nodes_out = [_serialize_node(root, hops=0, is_root=True)]

    elapsed = round((time.time() - t0) * 1000, 1)
    return {
        "graph_nodes": nodes_out,
        "graph_edges": edges_out,
        "raw_root_document": raw_root,
        "highlight_nodes": highlight,
        "steps": [_step(
            "mongo_executor", "mongo", "Execute aggregation on banking.network_graph",
            fn=node_mongo_executor,
            input={"action": action, "pipeline": pipeline,
                   "target": plan.get("target_entity"),
                   "show_all_relationships": show_all},
            output={
                "result_count": result_count,
                "rows_inspected": rows_inspected,
                "nodes_returned": len(nodes_out),
                "edges_returned": len(edges_out),
                "dangling_edges_kept": dangling_edges,
                "raw_root_captured": bool(raw_root),
                "highlight_count": len(highlight),
            },
            ms=elapsed,
        )],
    }


def node_narrative_writer(state: InvestigatorState, llm_url: str) -> dict:
    """LLM call: write an answer GROUNDED in the evidence package."""
    t0 = time.time()
    intent = state.get("intent") or {}
    nodes = state.get("graph_nodes") or []
    edges = state.get("graph_edges") or []
    evidence = state.get("evidence") or {}

    # Build subgraph summary for context (LLM can mention counts but only ones
    # already reflected in evidence.neighbour_summary).
    by_type: dict[str, int] = {}
    edge_types: dict[str, int] = {}
    for n in nodes:
        by_type[n.get("type", "?")] = by_type.get(n.get("type", "?"), 0) + 1
    for e in edges:
        edge_types[e.get("label", "?")] = edge_types.get(e.get("label", "?"), 0) + 1

    # Compact evidence package — keep pipelines as Python objects, not strings,
    # but cap depth to keep tokens manageable.
    compact_evidence = {
        "source_collection": evidence.get("source_collection"),
        "mongodb_operation": evidence.get("mongodb_operation"),
        "root_document_excerpt": evidence.get("root_document_excerpt"),
        "field_facts": evidence.get("field_facts") or [],
        "alert_provenance": [
            {
                "kind": a.get("kind"),
                "severity": a.get("severity"),
                "title": a.get("title"),
                "summary": a.get("summary"),
                "detection_pipeline_id": a.get("kind"),  # narrative refers by id, full pipeline shown in UI
            }
            for a in (evidence.get("alert_provenance") or [])
        ],
    }

    payload = {
        "user_question": state["query"],
        "intent": intent,
        "subgraph_summary": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": by_type,
            "edge_types": edge_types,
        },
        "evidence": compact_evidence,
        "evidence_is_empty": not (
            compact_evidence["field_facts"]
            or compact_evidence["alert_provenance"]
            or compact_evidence["root_document_excerpt"]
        ),
    }

    messages = [
        {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, default=str)[:11000]},
    ]

    try:
        raw = _llm_call(llm_url, messages, max_tokens=700)
    except requests.RequestException as e:
        raw = f"(LLM call failed: {e})"

    # If the LLM accidentally wraps in JSON, strip it.
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", txt, flags=re.MULTILINE).strip()
    if txt.startswith("{") and txt.endswith("}"):
        parsed = _parse_json_loose(txt) or {}
        txt = parsed.get("narrative") or parsed.get("answer") or txt

    elapsed = round((time.time() - t0) * 1000, 1)
    return {
        "narrative": txt,
        "steps": [_step(
            "narrative_writer", "llm",
            "Generate narrative grounded ONLY in evidence package",
            fn=node_narrative_writer,
            input={
                "payload_keys": list(payload.keys()),
                "field_facts_count": len(compact_evidence["field_facts"]),
                "alert_provenance_count": len(compact_evidence["alert_provenance"]),
                "evidence_is_empty": payload["evidence_is_empty"],
            },
            output={"narrative_chars": len(txt)},
            ms=elapsed,
        )],
    }


# ──────────────────────────────────────────────────────────
# Graph assembly
# ──────────────────────────────────────────────────────────

def _build_graph(graph: Collection, llm_url: str, db=None):
    """Compile the 6-node LangGraph state machine.

    `db` is the pymongo Database object — passed to evidence_builder so it can
    run corroborating queries across other collections (kyc_documents, customers,
    insurance_policies, transactions, …) to prove the watchlist/AML flags are
    not invented.
    """
    sg = StateGraph(InvestigatorState)

    sg.add_node("intent_classifier", lambda s: node_intent_classifier(s, llm_url))
    sg.add_node("entity_resolver",   node_entity_resolver)
    sg.add_node("pipeline_planner",  node_pipeline_planner)
    sg.add_node("mongo_executor",    lambda s: node_mongo_executor(s, graph))
    sg.add_node("evidence_builder",  lambda s: node_evidence_builder(s, db))
    sg.add_node("narrative_writer",  lambda s: node_narrative_writer(s, llm_url))

    sg.add_edge(START, "intent_classifier")
    sg.add_edge("intent_classifier", "entity_resolver")
    sg.add_edge("entity_resolver",   "pipeline_planner")
    sg.add_edge("pipeline_planner",  "mongo_executor")
    sg.add_edge("mongo_executor",    "evidence_builder")
    sg.add_edge("evidence_builder",  "narrative_writer")
    sg.add_edge("narrative_writer",  END)

    return sg.compile()


# ──────────────────────────────────────────────────────────
# Catalog & alerts helpers — duplicated lean from app.py
# (kept here so graph_agent.py is self-contained)
# ──────────────────────────────────────────────────────────

def build_catalog(graph: Collection, max_nodes: int = 400) -> tuple[str, set]:
    """Compact ID|TYPE|LABEL listing + a set of valid IDs for membership checks."""
    lines: list[str] = []
    ids: set[str] = set()
    cur = graph.find(
        {},
        {"_id": 1, "type": 1, "label": 1, "metadata": 1},
    ).limit(max_nodes)
    for d in cur:
        ids.add(d["_id"])
        m = d.get("metadata", {}) or {}
        meta_bits = []
        if m.get("watchlist"):           meta_bits.append("watchlist")
        if m.get("aml_flag"):            meta_bits.append("AML")
        if m.get("shared_by", 0) >= 2:   meta_bits.append(f"shared_by_{m['shared_by']}")
        if m.get("policy_type"):         meta_bits.append(m["policy_type"])
        if m.get("account_type"):        meta_bits.append(m["account_type"])
        if m.get("purpose"):             meta_bits.append(m["purpose"])
        meta_str = (" [" + ",".join(meta_bits) + "]") if meta_bits else ""
        lines.append(f"{d['_id']} | {d.get('type')} | {d.get('label','')}{meta_str}")
    return "\n".join(lines), ids


# ──────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────

def run_investigation(
    *,
    query: str,
    history: list[dict] | None,
    graph: Collection,
    llm_url: str,
    catalog: str,
    catalog_ids: set,
    alerts: list[dict],
    show_all_relationships: bool = False,
    db=None,
) -> dict:
    """Run the 6-node LangGraph and return a frontend-shaped response."""
    t0 = time.time()
    if db is None:
        db = graph.database
    app_graph = _build_graph(graph, llm_url, db=db)

    initial: InvestigatorState = {
        "query": query,
        "history": history or [],
        "catalog": catalog,
        "catalog_ids": catalog_ids,
        "alerts": alerts,
        "show_all_relationships": show_all_relationships,
        "steps": [],
    }

    final = app_graph.invoke(initial)
    total_ms = round((time.time() - t0) * 1000, 1)

    plan = final.get("plan", {}) or {}
    action = plan.get("action") or "none"
    pipeline_descriptions = {
        "graph_lookup": "MongoDB $graphLookup — multi-hop traversal of relationships[]",
        "find_one":    "MongoDB $match + $limit:1 — direct read of the source document",
    }

    return {
        "query": query,
        "narrative": final.get("narrative", ""),
        "intent": final.get("intent", {}),
        "entities": final.get("entities", []),
        "plan": plan,
        "pipeline_used": {
            "description": pipeline_descriptions.get(action, "no pipeline"),
            "stages": final.get("pipeline", []),
            "action": action,
        } if final.get("pipeline") else None,
        "nodes": final.get("graph_nodes", []),
        "edges": final.get("graph_edges", []),
        "highlight_nodes": final.get("highlight_nodes", []),
        "evidence": final.get("evidence", {}),
        "steps": final.get("steps", []),
        "total_ms": total_ms,
        "root": plan.get("target_entity"),
    }
