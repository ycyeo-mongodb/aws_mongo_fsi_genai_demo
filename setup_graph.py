#!/usr/bin/env python3
"""
Build the `network_graph` collection used for $graphLookup-powered
fraud detection, AML tracing, insurance beneficiary networks and
loan-guarantor concentration analysis.

The graph follows the maritime workshop pattern: a single self-referential
collection where each document is a node and outgoing edges are embedded
in `relationships[]` — making $graphLookup uniformly applicable to every
node type.

Usage:
    python setup_graph.py            # rebuild from scratch
    python setup_graph.py --insurance-only  # only refresh insurance_policies collection

Idempotent: drops and recreates `network_graph`. Insurance collection is also
dropped/recreated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DB_NAME = "banking"
GRAPH_COL = "network_graph"
INSURANCE_COL = "insurance_policies"


def normalize_label(s: str) -> str:
    """Canonical form used by the deduplicating compound index `(type, normalized_label)`.
    Lower-case, whitespace-collapsed, punctuation-tolerant — applied at WRITE time so
    queries can do an indexed equality match instead of an unsanchored $regex scan."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def load_json(name: str) -> Any:
    path = DATA_DIR / name
    if not path.is_file():
        raise SystemExit(f"Missing data file: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def cust_label(cust: dict) -> str:
    return cust.get("full_name") or cust.get("name") or cust.get("id", "")


def acct_label(acct: dict) -> str:
    type_map = {"savings": "Savings", "current": "Current", "fixed_deposit": "Fixed Deposit"}
    t = type_map.get(acct.get("account_type", ""), "Account")
    last4 = (acct.get("account_number") or "").split("-")[-1]
    return f"{t} •••{last4}" if last4 else t


def loan_label(loan: dict) -> str:
    purpose = (loan.get("purpose") or "").title()
    amt = loan.get("amount_usd") or loan.get("loan_amount") or 0
    return f"{purpose} Loan ${amt:,.0f}" if amt else f"{purpose} Loan"


def policy_label(pol: dict) -> str:
    return f"{pol.get('policy_type','').title()} Policy ${pol.get('coverage_usd',0):,.0f}"


# ──────────────────────────────────────────────────────────
# Counterparty extraction from transaction descriptions
# ──────────────────────────────────────────────────────────

def extract_counterparty(description: str) -> str | None:
    """Pull a merchant/counterparty name out of a free-text transaction description."""
    if not description:
        return None
    desc = description.strip()
    # Common patterns: "Salary — AnyCompany", "Payment to GroceryMart", "Transfer to ACC-xxxxx"
    patterns = [
        r"^(?:Salary|Payment)\s*(?:from|to|—|-|–)\s*([A-Z][A-Za-z &']+)",
        r"^(?:Transfer|Wire|Remit)\s*(?:to|from|—|-)\s*([A-Z][A-Za-z &']+)",
        r"^([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,3})\s*(?:purchase|charge|payment)?$",
    ]
    for pat in patterns:
        m = re.search(pat, desc)
        if m:
            name = m.group(1).strip()
            if 3 <= len(name) <= 40:
                return name
    return None


# ──────────────────────────────────────────────────────────
# Insurance policies → MongoDB
# ──────────────────────────────────────────────────────────

def load_insurance(db) -> list[dict]:
    print("\n" + "=" * 60)
    print(f"STEP 1: Load insurance policies → banking.{INSURANCE_COL}")
    print("=" * 60)
    policies = load_json("insurance_policies.json")
    db[INSURANCE_COL].drop()

    # map customer string IDs (CUST-xxxxx) to ObjectId for cross-collection joins
    cust_id_to_oid = {c["id"]: c["_id"] for c in db.customers.find({}, {"id": 1})}

    docs = []
    for p in policies:
        holder_str = p.get("holder_id", "")
        beneficiaries_str = p.get("beneficiary_ids", []) or []
        doc = {
            **p,
            "holder_oid": cust_id_to_oid.get(holder_str),
            "beneficiary_oids": [cust_id_to_oid.get(b) for b in beneficiaries_str if cust_id_to_oid.get(b)],
        }
        docs.append(doc)

    if docs:
        db[INSURANCE_COL].insert_many(docs)
    print(f"  ✓ Inserted {db[INSURANCE_COL].count_documents({})} policies")
    return policies


# ──────────────────────────────────────────────────────────
# Build the unified network_graph collection
# ──────────────────────────────────────────────────────────

def build_graph(db, policies: list[dict], seeds: dict) -> None:
    print("\n" + "=" * 60)
    print(f"STEP 2: Build network_graph → banking.{GRAPH_COL}")
    print("=" * 60)

    # Pull source data
    customers = list(db.customers.find({}))
    accounts = list(db.accounts.find({}))
    loans = list(db.loan_applications.find({}))
    transactions = list(db.transactions.find({}))

    print(f"  source: {len(customers)} customers, {len(accounts)} accounts, "
          f"{len(loans)} loans, {len(transactions)} transactions, "
          f"{len(policies)} policies")

    # Lookup helpers — string id → record
    cust_by_id = {c.get("id"): c for c in customers}
    acct_by_id = {a.get("id"): a for a in accounts}
    loan_by_id = {l.get("id"): l for l in loans}

    # nodes: dict keyed by graph _id → node document under construction
    nodes: dict[str, dict] = {}

    def add_node(node_id: str, **fields) -> dict:
        if node_id not in nodes:
            label = fields.get("label", node_id)
            nodes[node_id] = {
                "_id": node_id,
                "type": fields.get("type", "Unknown"),
                "label": label,
                "normalized_label": normalize_label(label),  # for indexed equality lookups
                "metadata": fields.get("metadata", {}),
                "relationships": [],
                "source_documents": [],
            }
        else:
            if fields.get("label"):
                nodes[node_id]["label"] = fields["label"]
                nodes[node_id]["normalized_label"] = normalize_label(fields["label"])
            if fields.get("metadata"):
                nodes[node_id]["metadata"].update(fields["metadata"])
            if fields.get("type"):
                nodes[node_id]["type"] = fields["type"]
        return nodes[node_id]

    def add_edge(src_id: str, target_id: str, edge_type: str,
                 source: str = "", **edge_meta) -> None:
        """source: free-text provenance string (which file/record produced this edge)."""
        if src_id not in nodes:
            return
        rel = {"target": target_id, "type": edge_type}
        if source:
            rel["source"] = source
        if edge_meta:
            rel.update(edge_meta)
        nodes[src_id]["relationships"].append(rel)

    def add_source(node_id: str, doc_id: str) -> None:
        if node_id in nodes and doc_id not in nodes[node_id]["source_documents"]:
            nodes[node_id]["source_documents"].append(doc_id)

    # 1) Customer nodes
    watchlist = set(seeds.get("watchlist", []))
    for c in customers:
        cid = c.get("id")
        meta = {
            "credit_score": c.get("credit_score"),
            "monthly_income_usd": c.get("monthly_income") or c.get("income_usd"),
            "employment_type": c.get("employment_type"),
            "risk_level": c.get("risk_level"),
            "watchlist": cid in watchlist,
        }
        add_node(cid, type="Customer", label=cust_label(c), metadata=meta)

    # 2) Account nodes + Customer→Account edges
    for a in accounts:
        aid = a.get("id")
        cid_ref = a.get("customer_id")
        # customer_id is now an ObjectId in the live collection — find string id via reverse lookup
        cust_str = next((cs for cs, c in cust_by_id.items() if c.get("_id") == cid_ref), None)
        meta = {
            "account_type": a.get("account_type"),
            "balance_usd": a.get("balance_usd"),
            "account_number": a.get("account_number"),
            "status": a.get("status"),
        }
        add_node(aid, type="Account", label=acct_label(a), metadata=meta)
        if cust_str:
            src = f"data/accounts.json:{aid} (customer_id field)"
            add_edge(cust_str, aid, "OWNS_ACCOUNT", source=src)
            add_edge(aid, cust_str, "OWNED_BY", source=src)

    # 3) Loan nodes + Customer→Loan edges
    for l in loans:
        lid = l.get("id")
        cid_ref = l.get("customer_id")
        cust_str = next((cs for cs, c in cust_by_id.items() if c.get("_id") == cid_ref), None)
        meta = {
            "amount_usd": l.get("amount_usd") or l.get("loan_amount"),
            "purpose": l.get("purpose"),
            "term_months": l.get("term_months"),
            "status": l.get("status"),
            "monthly_payment_usd": l.get("monthly_payment_usd") or l.get("monthly_payment"),
        }
        add_node(lid, type="Loan", label=loan_label(l), metadata=meta)
        if cust_str:
            src = f"data/loan_applications.json:{lid} (customer_id field)"
            add_edge(cust_str, lid, "APPLIED_FOR", source=src)
            add_edge(lid, cust_str, "APPLIED_BY", source=src)

    # 4) Insurance Policy nodes + holder/beneficiary edges
    for p in policies:
        pid = p["id"]
        meta = {
            "policy_type": p.get("policy_type"),
            "coverage_usd": p.get("coverage_usd"),
            "premium_monthly_usd": p.get("premium_monthly_usd"),
            "status": p.get("status"),
            "underwriter": p.get("underwriter"),
            "issue_date": p.get("issue_date"),
            "expiry_date": p.get("expiry_date"),
        }
        add_node(pid, type="InsurancePolicy", label=policy_label(p), metadata=meta)
        holder = p.get("holder_id")
        if holder and holder in nodes:
            src_h = f"data/insurance_policies.json:{pid} (holder_id)"
            add_edge(holder, pid, "POLICY_HOLDER", source=src_h)
            add_edge(pid, holder, "HELD_BY", source=src_h)
        for b in p.get("beneficiary_ids", []):
            if b in nodes:
                src_b = f"data/insurance_policies.json:{pid} (beneficiary_ids)"
                add_edge(b, pid, "BENEFICIARY_OF", source=src_b)
                add_edge(pid, b, "BENEFICIARY_IS", source=src_b)

    # 5) Shared attributes (addresses, phones, devices, employers)
    for grp in seeds.get("shared_addresses", []):
        nid = grp["address_id"]
        add_node(nid, type="Address", label=grp["address_text"],
                 metadata={"shared_by": len(grp["customers"])})
        for cust in grp["customers"]:
            if cust in nodes:
                src = f"data/graph_seeds.json:shared_addresses[{nid}] — KYC address-on-file"
                add_edge(cust, nid, "LIVES_AT", source=src)
                add_edge(nid, cust, "RESIDENT", source=src)

    for grp in seeds.get("shared_phones", []):
        nid = grp["phone_id"]
        add_node(nid, type="Phone", label=grp["phone_text"],
                 metadata={"shared_by": len(grp["customers"])})
        for cust in grp["customers"]:
            if cust in nodes:
                src = f"data/graph_seeds.json:shared_phones[{nid}] — customer profile contact"
                add_edge(cust, nid, "USES_PHONE", source=src)
                add_edge(nid, cust, "PHONE_OF", source=src)

    for grp in seeds.get("shared_devices", []):
        nid = grp["device_id"]
        add_node(nid, type="Device", label=grp.get("device_label", nid),
                 metadata={"shared_by": len(grp["customers"])})
        for cust in grp["customers"]:
            if cust in nodes:
                src = f"data/graph_seeds.json:shared_devices[{nid}] — login fingerprint"
                add_edge(cust, nid, "USES_DEVICE", source=src)
                add_edge(nid, cust, "DEVICE_OF", source=src)

    for grp in seeds.get("shared_employers", []):
        nid = grp["employer_id"]
        add_node(nid, type="Employer", label=grp["employer_name"],
                 metadata={"shared_by": len(grp["customers"])})
        for cust in grp["customers"]:
            if cust in nodes:
                src = f"data/graph_seeds.json:shared_employers[{nid}] — payroll record"
                add_edge(cust, nid, "WORKS_AT", source=src)
                add_edge(nid, cust, "EMPLOYS", source=src)

    # 6) Counterparty nodes + edges from real transactions (aggregated)
    cp_pair_count: Counter = Counter()
    cp_pair_total: defaultdict = defaultdict(float)
    cp_pair_acct: dict = {}

    for t in transactions:
        cp_name = extract_counterparty(t.get("description", ""))
        if not cp_name:
            continue
        acct_oid = t.get("account_id")
        # find string acct id via reverse lookup
        acct_str = next((s for s, a in acct_by_id.items() if a.get("_id") == acct_oid), None)
        if not acct_str:
            continue
        cp_id = "CP-" + re.sub(r"[^A-Za-z0-9]", "", cp_name)[:24]
        key = (acct_str, cp_id)
        cp_pair_count[key] += 1
        cp_pair_total[key] += float(t.get("amount_usd", 0) or 0)
        cp_pair_acct[cp_id] = cp_name

    for cp_id, name in cp_pair_acct.items():
        add_node(cp_id, type="Counterparty", label=name,
                 metadata={"category": "merchant_or_employer"})

    for (acct_str, cp_id), count in cp_pair_count.items():
        if count < 2:
            continue  # skip one-offs to keep visualization readable
        total = cp_pair_total[(acct_str, cp_id)]
        src = f"data/transactions.json — aggregated from {count} transactions"
        add_edge(acct_str, cp_id, "TRANSACTED_WITH",
                 source=src, transaction_count=count, total_amount_usd=round(total, 2))
        add_edge(cp_id, acct_str, "RECEIVED_FROM",
                 source=src, transaction_count=count, total_amount_usd=round(total, 2))

    # 7) AML chain — explicit account-to-account wire transfers
    for i, hop in enumerate(seeds.get("aml_chain", [])):
        from_acct = hop["from_account"]
        to_acct = hop["to_account"]
        if from_acct in nodes and to_acct in nodes:
            src = f"data/graph_seeds.json:aml_chain[step {i+1}] — {hop['date']}"
            add_edge(from_acct, to_acct, "WIRE_SENT", source=src,
                     amount_usd=hop["amount_usd"], date=hop["date"],
                     description=hop["description"], chain_step=i + 1)
            add_edge(to_acct, from_acct, "WIRE_RECEIVED", source=src,
                     amount_usd=hop["amount_usd"], date=hop["date"],
                     description=hop["description"], chain_step=i + 1)
            # boost the participating customers as flagged
            from_cust = hop["from_customer"]
            to_cust = hop["to_customer"]
            if from_cust in nodes:
                nodes[from_cust]["metadata"]["aml_flag"] = True
            if to_cust in nodes:
                nodes[to_cust]["metadata"]["aml_flag"] = True

    # 8) Guarantors
    for g in seeds.get("guarantors", []):
        guar = g["guarantor"]
        loan_id = g["loan_id"]
        if guar in nodes and loan_id in nodes:
            src = f"data/graph_seeds.json:guarantors — loan {loan_id}"
            add_edge(guar, loan_id, "GUARANTOR_OF", source=src)
            add_edge(loan_id, guar, "GUARANTEED_BY", source=src)

    # 9) Co-applicants
    for c in seeds.get("co_applicants", []):
        primary = c["primary_id"]
        coapp = c["co_applicant_id"]
        loan_id = c["loan_id"]
        if primary in nodes and coapp in nodes:
            src = f"data/graph_seeds.json:co_applicants — loan {loan_id}"
            add_edge(primary, coapp, "CO_APPLICANT_WITH", source=src, loan_id=loan_id)
            add_edge(coapp, primary, "CO_APPLICANT_WITH", source=src, loan_id=loan_id)
        if coapp in nodes and loan_id in nodes:
            add_edge(coapp, loan_id, "CO_APPLIED_FOR", source=src)

    # ─── Insert into MongoDB ───
    db[GRAPH_COL].drop()
    docs_to_insert = list(nodes.values())
    if docs_to_insert:
        db[GRAPH_COL].insert_many(docs_to_insert)

    # ─── Indexes (critical for $graphLookup performance + indexed lookups) ───
    db[GRAPH_COL].create_index("type")
    db[GRAPH_COL].create_index("relationships.target")
    db[GRAPH_COL].create_index("metadata.watchlist")
    db[GRAPH_COL].create_index("metadata.aml_flag")
    # Compound index supports the entity-extraction equality match
    # `{type: ..., normalized_label: ...}` used by the document-onboarding pipeline.
    db[GRAPH_COL].create_index([("type", 1), ("normalized_label", 1)],
                               name="type_1_normalized_label_1")

    # ─── Stats ───
    type_counts = Counter(n["type"] for n in docs_to_insert)
    edge_count = sum(len(n["relationships"]) for n in docs_to_insert)

    print(f"\n  ✓ Inserted {len(docs_to_insert)} nodes:")
    for typ, count in type_counts.most_common():
        print(f"      {typ:<18} {count:>5}")
    print(f"  ✓ Total edges (sum of relationships[]): {edge_count}")
    print("  ✓ Indexes: type, relationships.target, metadata.watchlist,")
    print("           metadata.aml_flag, (type, normalized_label) compound")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--insurance-only", action="store_true",
                        help="Only refresh insurance_policies (skip graph rebuild)")
    args = parser.parse_args()

    load_dotenv(SCRIPT_DIR / ".env")
    import os
    mongodb_uri = os.environ.get("MONGODB_URI", "").strip()
    if not mongodb_uri:
        print("ERROR: MONGODB_URI not set", file=sys.stderr)
        sys.exit(1)

    print("Connecting to MongoDB Atlas ...")
    try:
        mongo = MongoClient(mongodb_uri, serverSelectionTimeoutMS=15000)
        mongo.admin.command("ping")
    except PyMongoError as e:
        print(f"ERROR: Could not connect to MongoDB: {e}", file=sys.stderr)
        sys.exit(1)
    print("  Connected.")

    db = mongo[DB_NAME]

    # Verify prerequisites
    if db.customers.count_documents({}) == 0:
        print("ERROR: customers collection is empty. Run setup_workshop.py first.", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    policies = load_insurance(db)
    if not args.insurance_only:
        seeds = load_json("graph_seeds.json")
        build_graph(db, policies, seeds)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    mongo.close()


if __name__ == "__main__":
    main()
