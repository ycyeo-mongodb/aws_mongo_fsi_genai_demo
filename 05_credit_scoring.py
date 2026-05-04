#!/usr/bin/env python3
"""
Workshop 05: Credit scoring pipeline — MongoDB profile + rule-based risk + LLM explanation (via API Gateway).
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests
from pymongo import MongoClient
from pymongo.database import Database

SCRIPT_DIR = Path(__file__).resolve().parent


def get_customer_profile(db: Database, customer_id: str) -> dict[str, Any] | None:
    """Join customer with all their loan applications via $lookup."""
    pipeline = [
        {"$match": {"id": customer_id}},
        {
            "$lookup": {
                "from": "loan_applications",
                "localField": "id",
                "foreignField": "customer_id",
                "as": "loan_applications",
            }
        },
    ]
    rows = list(db.customers.aggregate(pipeline))
    return rows[0] if rows else None


def _payment_history_risk(payment_history: str) -> float:
    order = {"excellent": 0.0, "good": 0.25, "fair": 0.55, "poor": 0.85}
    return order.get((payment_history or "").lower(), 0.5)


def _employment_risk(employment_type: str) -> float:
    """Higher sub-score = riskier (0–1)."""
    t = (employment_type or "").lower()
    risky = {"farmer": 0.35, "self_employed": 0.3, "business_owner": 0.2}
    stable = {"government": 0.05, "salaried": 0.08}
    return risky.get(t, stable.get(t, 0.22))


def compute_risk_score(profile: dict[str, Any]) -> dict[str, Any]:
    """
    Rule-based risk score 0–100 (higher = riskier).
    Uses: debt_to_income_ratio (max across linked applications), payment_history, credit_score,
    account_age_months, employment_type.
    """
    c = profile
    apps = profile.get("loan_applications") or []
    dti = max((float(x.get("debt_to_income_ratio") or 0) for x in apps), default=0.0)

    credit = float(c.get("credit_score") or 500)
    acct_age = float(c.get("account_age_months") or 0)

    # Component contributions (each roughly 0–1), then scaled to 0–100
    dti_part = min(1.0, max(0.0, dti) / 0.65)
    credit_part = 1.0 - min(1.0, max(0.0, (credit - 300) / 550))
    age_part = 1.0 - min(1.0, acct_age / 120.0)
    pay_part = _payment_history_risk(str(c.get("payment_history", "")))
    emp_part = _employment_risk(str(c.get("employment_type", "")))

    raw = (
        0.28 * dti_part
        + 0.22 * credit_part
        + 0.18 * age_part
        + 0.20 * pay_part
        + 0.12 * emp_part
    )
    score = round(min(100.0, max(0.0, raw * 100.0)), 1)

    if score < 40:
        risk_level = "low"
    elif score < 70:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "inputs": {
            "debt_to_income_ratio": dti,
            "credit_score": credit,
            "account_age_months": acct_age,
            "payment_history": c.get("payment_history"),
            "employment_type": c.get("employment_type"),
        },
    }


def generate_explanation(
    llm_url: str,
    customer: dict[str, Any],
    risk_result: dict[str, Any],
    loan_app: dict[str, Any],
) -> str:
    """Call the LLM API Gateway for approval/decline narrative."""
    system = (
        "You are a credit risk analyst at a bank. Explain the credit decision in clear, "
        "professional language. Mention specific factors."
    )
    payload = {
        "customer_summary": {
            "id": customer.get("id"),
            "name": customer.get("name"),
            "employment_type": customer.get("employment_type"),
            "income_usd": customer.get("income_usd"),
            "credit_score": customer.get("credit_score"),
            "payment_history": customer.get("payment_history"),
            "account_age_months": customer.get("account_age_months"),
        },
        "risk_assessment": risk_result,
        "loan_application": {
            "id": loan_app.get("id"),
            "amount_usd": loan_app.get("amount_usd"),
            "purpose": loan_app.get("purpose"),
            "term_months": loan_app.get("term_months"),
            "debt_to_income_ratio": loan_app.get("debt_to_income_ratio"),
            "monthly_payment_usd": loan_app.get("monthly_payment_usd"),
            "status": loan_app.get("status"),
        },
    }
    user = (
        "Based on the structured data below, respond with:\n"
        "1) A single word decision line: DECISION: APPROVE or DECISION: DECLINE\n"
        "2) A short paragraph explaining why, referencing the numbers and factors given.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
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


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")
    uri = os.environ.get("MONGODB_URI")
    llm_url = os.environ.get("LLM_API_URL")
    if not uri or not llm_url:
        raise SystemExit("Set MONGODB_URI and LLM_API_URL in .env")

    mongo = MongoClient(uri)
    db = mongo["banking"]

    # First five loan applications as demo cases
    sample_loans = list(db.loan_applications.find().sort("id", 1).limit(5))

    print("=== Credit scoring demo (5 sample applications) ===\n")

    for loan in sample_loans:
        cid = loan.get("customer_id")
        profile = get_customer_profile(db, cid)
        if not profile:
            print(f"[SKIP] No customer for loan {loan.get('id')} / customer_id={cid}\n")
            continue

        customer_core = {k: v for k, v in profile.items() if k != "loan_applications"}
        # Scope applications to this loan so DTI aligns with the case being scored
        scoped_profile = {**profile, "loan_applications": [loan]}
        risk = compute_risk_score(scoped_profile)
        explanation = generate_explanation(llm_url, customer_core, risk, loan)

        print("—" * 60)
        print(f"Loan:           {loan.get('id')}  |  Customer: {customer_core.get('name')} ({cid})")
        amt = float(loan.get("amount_usd") or 0)
        print(f"Amount:         ${amt:,.2f}  |  Purpose: {loan.get('purpose')}")
        print(f"DTI:            {loan.get('debt_to_income_ratio')}")
        print(f"Risk score:     {risk['risk_score']}  |  Risk level: {risk['risk_level']}")
        print()
        print("Model decision & explanation:")
        print(explanation)
        print()

    mongo.close()
    print("Done.")


if __name__ == "__main__":
    main()
