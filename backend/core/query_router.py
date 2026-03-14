"""
query_router.py — Text normalisation utilities and LLM-based query routing.

Functions
---------
normalize_numbers(s)
    Lower-case + strip currency symbols, commas, and non-breaking spaces.

bm25_tokenize(s)
    Tokenise text for BM25, preserving decimal numbers and percentages.

predict_query_document_type(query, available_types)
    Use the Gemma LLM to predict which document type most likely contains
    the answer for a given query.  Returns (predicted_type, confidence).
"""

import re
import json
from typing import List, Tuple

from llm.llm_router import llm as gemma_llm


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def normalize_numbers(s: str) -> str:
    """
    Normalise a string for consistent BM25 matching.

    Transformations applied:
      - lower-case
      - remove commas  (380,000 → 380000)
      - remove dollar signs ($380 → 380)
      - replace non-breaking spaces (U+00A0) with regular spaces
      - collapse repeated whitespace

    Args:
        s: input string

    Returns:
        Normalised string.
    """
    s = s.lower()
    s = s.replace(",", "")       # 380,000 → 380000
    s = s.replace("$", "")       # $380 → 380
    s = s.replace("\u00a0", " ") # NBSP safety
    s = re.sub(r"\s+", " ", s).strip()
    return s


def bm25_tokenize(s: str) -> List[str]:
    """
    Tokenise a string for BM25, preserving decimal numbers and percentages.

    Extracts:
      - alphabetic words
      - integers, decimals, percentages  (e.g. 3.75, 12.5%)

    Args:
        s: raw text (query or document chunk)

    Returns:
        List of tokens.
    """
    s = normalize_numbers(s)
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?%?", s)


# ---------------------------------------------------------------------------
# LLM-based query routing
# ---------------------------------------------------------------------------

# Keyword hints used to steer the router prompt
_KEYWORD_HINTS = {
    "Resume": (
        "career, experience, education, skills, employment history, "
        "qualifications, work history, objective, summary, functional resume, "
        "GPA, university"
    ),
    "Lender Fee Sheet": (
        "loan estimate, loan amount, property owners, borrowers, loan term, "
        "loan purpose, closing disclosure, interest rate, sale price, "
        "cash to close, projected payments, APR, annual percentage rate, "
        "origination charges, services you cannot shop for, lender credits, "
        "estimated closing costs, prepaid interest, escrow, rate lock, "
        "fees worksheet, total monthly payment, closing cost details"
    ),
    "Mortgage": (
        "mortgage document, FHA mortgage, Wisconsin mortgage, security "
        "instrument, mortgagor, borrower on mortgage, MERS, MERS phone, "
        "MERS address, MIN number, recording date mortgage, register of deeds, "
        "lender address on mortgage, parcel identifier, FHA case number, "
        "county recorded mortgage, section township, document number mortgage"
    ),
    "Discount Notice": (
        "CA discount notice, notice of available discounts, fee reduction, "
        "settlement program, disaster loans discount, churches discount, "
        "charitable discount, employee rate, preliminary report reopened, "
        "qualifying period, $20 discount, lender's policy discount"
    ),
    "Contract": (
        "employment agreement, contract of employment, obligations, parties, "
        "legal terms, signing, clauses, indemnify, governing law, termination, "
        "confidentiality, probationary period, advance notice, working hours, "
        "overtime rate, PPE, protective equipment, severability, assignment"
    ),
    "Pay Slip": (
        "salary, wages, deductions, net pay, gross pay, pay period, YTD, "
        "total earnings, total deductions, employee id, W-2, federal income "
        "tax withheld, social security wages, payslip, payworks, basic pay, "
        "allowance, overtime, provident fund, professional tax, CPP, EI, "
        "life insurance benefit, dental benefit, home hourly rate"
    ),
    "Bank Statement": (
        "account balance, transactions, deposits, withdrawals, beginning "
        "balance, ending balance, statement period, posting date, ACH deposit, "
        "check deposit, FDIC insured, electronic deposits, electronic payments, "
        "other withdrawals, average collected balance, counter deposit"
    ),
    "Preliminary Title Report": (
        "preliminary report, title report, vesting, legal description, wire "
        "instructions, routing number, ABA, swift, beneficiary, APN, deed of "
        "trust, trustee, easement, exceptions to coverage, vested in, joint "
        "tenants, recording no, CLTA, ALTA, title officer, order number, "
        "property tax installment, CC&Rs, credit line closure, "
        "statement of information, transmittal, reconveyance"
    ),
    "Tax Document": (
        "property tax, tax bill, tax return, tax form, installment, 1099, "
        "fiscal year, assessed value, delinquent, penalty, exemption, "
        "treasurer-tax collector, code area"
    ),
    "Privacy Statement": (
        "privacy statement, privacy notice, personal information, non-public "
        "personal information, disclosure of personal information, affiliated "
        "companies, nonaffiliated third parties, privacy practices, opt-out, "
        "information we collect, information we share"
    ),
    "Invoice": (
        "invoice, bill to, ship to, quantity, unit price, subtotal, total due, "
        "invoice number, balance due, tax rate, due on receipt"
    ),
    "Insurance": (
        "coverage, policy, premiums, deductible, limits, declarations, insured, "
        "exclusions, endorsements, schedule A, schedule B, covered risks, "
        "ALTA homeowner's policy, maximum dollar limit, deductible amount"
    ),
    "Letter": (
        "dear, sincerely, to whom it may concern, regards, correspondence, "
        "enclosed please find"
    ),
    "ID Document": (
        "driver license, passport, DOB, date of birth, expiration, issue date, "
        "LIC NO, sex, height, weight, eyes, class, endorsements, "
        "Kansas driver's license, restrictions, license number, license class"
    ),
    "Other": "general, unclear, miscellaneous",
}


def predict_query_document_type(
    query: str,
    available_types: List[str],
) -> Tuple[str, float]:
    """
    Predict which document type in the index most likely contains the answer.

    Args:
        query          : user's natural-language question
        available_types: document types actually present in the current index

    Returns:
        (predicted_type, confidence)
        - predicted_type: one of available_types (or 'Other')
        - confidence    : float in [0.0, 1.0]
    """
    available_hints = "\n".join(
        f"- {doc_type}: {_KEYWORD_HINTS.get(doc_type, 'General content')}"
        for doc_type in available_types
    )

    prompt = f"""
        Analyze this query and predict which document type would most likely contain the answer.

        Query: {query}

        Choose the MOST LIKELY type from this given options strictly.
        Available document types in this PDF:
        {available_types}

        Keyword hints for each document type:
        {available_hints}

        IMPORTANT ROUTING RULES:
        - If the query mentions "Wisconsin Mortgage", "FHA Mortgage", "mortgage document", "mortgagor", "MERS" on a mortgage, "register of deeds", or "recording" of a mortgage, route to "Mortgage".
        - If the query mentions "CA Discount Notice", "Notice of Available Discounts", "discount for churches", "disaster loans discount", "fee reduction settlement", "$20 discount", or "qualifying period", route to "Discount Notice".
        - If the query mentions "Credit Line Closure Request" or "reconveyance documents", route to "Preliminary Title Report".
        - If the query mentions "Loan Estimate", "Fees Worksheet", "closing costs", "origination charges", route to "Lender Fee Sheet".
        - Do NOT route mortgage questions to "Lender Fee Sheet" - mortgages are separate documents.
        - Do NOT route discount notice questions to "Insurance" or "Lender Fee Sheet".

        If the query doesn't clearly match any type, respond with "Other".

        Respond in JSON format only:
        IMPORTANT: Do not use leading zeros in numbers (use 0.85, not 00.85)
        {{"type": "DocumentType", "confidence": 0.85}}

        Confidence should be between 0.0 and 1.0
    """

    try:
        response = gemma_llm.complete(prompt, temperature=0)
        print(f"🤖 LLM Response: {response}")

        json_match = re.search(r"\{[^{}]*\}", response.text)
        if not json_match:
            raise ValueError("No JSON object found in LLM response")

        json_str = json_match.group()
        print(f"📝 Parsed JSON: {json_str}")

        result = json.loads(json_str.strip())
        return result.get("type", "Other"), float(result.get("confidence", 0.5))

    except Exception as e:
        print(f"❌ Query routing error: {e}")
        return "Other", 0.0
