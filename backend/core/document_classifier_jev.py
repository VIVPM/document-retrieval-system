"""Jev document type classification and boundary detection for PDF ingestion."""

import math
import os
from functools import lru_cache
from pathlib import Path

import requests
from dotenv import dotenv_values

from logging_setup import get_logger

log = get_logger("drs.classifier")
MODEL = "jev-1.13.0"
URL = "https://api.typesafe.ai/v1/systemone"

VALID_DOC_TYPES = {
    "Loan Application": "Uniform Residential Loan Application (URLA/Form 1003) or lender loan application requesting loan amount, property, purpose, employment, income, assets and declarations. Not a title-company Statement of Information for a title order.",
    "Lender Fee Sheet": "Loan Estimate, Closing Disclosure or Fees Worksheet stating loan terms, payments, APR, origination fees and cash to close. Not the borrower's application.",
    "Pay Slip": "Earnings statement for a pay period, with gross or net pay, deductions and employee details. Not an annual W-2.",
    "Bank Statement": "Bank or credit-union account statement with period, deposits, withdrawals, balances and account activity.",
    "Tax Document": "W-2, 1099, tax return or property tax bill/statement; annual wages or tax information.",
    "Preliminary Title Report": "Title company's preliminary report with vesting, legal description, APN, recorded liens and easements. Also title-package forms headed Statement of Information, Affidavit, Certification of Trust, Credit Line or Equity Line Closure Request, or TRANSMITTAL, even when they collect personal details. Not a policy jacket of covered risks/exclusions.",
    "Mortgage": "The recorded mortgage or deed-of-trust security instrument itself, with borrowers, lender, covenants and signatures. Not a title report listing a deed of trust.",
    "Insurance": "Insurance policy terms or title insurance policy jacket with covered risks, exclusions, conditions and endorsements. Not a preliminary title report listing policies to be issued.",
    "Discount Notice": "Title or escrow company's notice or request form for available discounts, including settlement, disaster or employee rates.",
    "Privacy Statement": "Company privacy notice about collection, sharing, affiliates and opt-out of personal information.",
    "Contract": "Agreement between parties, including employment, purchase or service contract, but not a mortgage security instrument.",
    "Letter": "Standalone borrower or third-party correspondence, gift letter, explanation, verification letter or memo; not a title company's credit-line closure form or transmittal.",
    "ID Document": "Driver's license, passport or other identity document.",
    "Resume": "Curriculum vitae, professional profile, education and work history.",
    "Invoice": "Bill with items, amounts, subtotal or amount due.",
    "Other": "Page not fitting the above types, including blank, unreadable or unrelated material.",
}


@lru_cache(maxsize=1)
def _api_key() -> str:
    key = os.getenv("TYPESAFE_API_KEY") or dotenv_values(
        Path(__file__).resolve().parents[1] / ".env").get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is required for Jev ingestion. Set it in backend/.env.")
    return key


def _ask(state, question: dict, kind: str) -> dict:
    response = requests.post(
        URL,
        headers={"Authorization": f"Bearer {_api_key()}"},
        json={"model": MODEL, "state": state, "questions": {kind: question}},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage") or {}
    tokens = usage.get("input_tokens")
    log.info("jev call", extra={"decision_type": kind, "jev_model": data.get("model"),
                                "input_tokens": tokens,
                                "cost_usd": round(tokens * 0.042 / 1_000_000, 8)
                                if isinstance(tokens, int) else None})
    return data["answers"][kind]


def classify_document_type(text: str, max_length: int = 1500) -> str:
    """Choose a document type for one page; provider failures retry the job."""
    if not (text or "").strip():
        return "Other"
    answer = _ask(text[:max_length], {
        "type": "choice",
        "instructions": "What kind of document is this page? Decide from its title and structure, not incidental references to other documents or amounts.",
        "criteria": VALID_DOC_TYPES,
    }, "doc_type")
    choice = answer.get("choice")
    if choice not in VALID_DOC_TYPES:
        log.warning("jev returned an unknown document type", extra={"reply": str(choice)[:80]})
        return "Other"
    return choice


def detect_document_boundary(prev_text: str, curr_text: str,
                             current_doc_type: str | None = None) -> bool:
    """True only when adjacent pages belong to the same physical document."""
    if not prev_text or not curr_text:
        return False
    state = {"previous_page_end": prev_text[-1500:],
             "current_page_start": curr_text[:1500],
             "previous_document_type": current_doc_type or "Unknown"}
    answer = _ask(state, {
        "type": "noul",
        "instructions": "Are these consecutive pages of the SAME physical document? Shared names, property, lender or loan amount alone do not make them one document. A new title, pay period, statement month, tax year, policy form or report begins a different document. A title transmittal followed by an 'Attachment One' policy-exclusions jacket begins a different document even if bundled together. Continuation headings and sequential sections indicate the same document.",
    }, "same_document")
    probability = answer.get("noul")
    if isinstance(probability, (int, float)) and math.isfinite(probability) and 0 <= probability <= 1:
        return probability >= 0.5
    log.warning("jev boundary response unparseable; keeping pages together",
                extra={"reply": str(probability)[:80]})
    return True


if __name__ == "__main__":
    from unittest.mock import patch

    assert classify_document_type("   ") == "Other"
    assert detect_document_boundary("", "text") is False
    with patch(__name__ + "._ask", side_effect=[
        {"choice": "Loan Application"}, {"choice": "unknown"},
        {"noul": 1.0}, {"noul": 0.0}, {"noul": None},
    ]):
        assert classify_document_type("borrower application") == "Loan Application"
        assert classify_document_type("unknown page") == "Other"
        assert detect_document_boundary("a", "b") is True
        assert detect_document_boundary("a", "b") is False
        assert detect_document_boundary("a", "b") is True
    print("Jev classifier selftest OK")
