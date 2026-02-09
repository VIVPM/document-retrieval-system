"""
document_classifier.py — LLM-based document type classification and
                          boundary detection between consecutive pages.

Functions
---------
classify_document_type(text, max_length)
    Classify a page/document into one of 14 pre-defined financial document
    categories using the Gemma LLM.

detect_document_boundary(prev_text, curr_text, current_doc_type)
    Decide whether two consecutive pages belong to the same logical document.
"""

from llm.llm_router import llm as gemma_llm

# ---------------------------------------------------------------------------
# Valid document categories
# ---------------------------------------------------------------------------

VALID_DOC_TYPES = [
    "Resume",
    "Mortgage",
    "Discount Notice",
    "Contract",
    "Pay Slip",
    "Bank Statement",
    "Tax Document",
    "Insurance",
    "Letter",
    "ID Document",
    "Privacy Statement",
    "Lender Fee Sheet",
    "Preliminary Title Report",
    "Invoice",
    "Other",
]

# ---------------------------------------------------------------------------
# Document type classification
# ---------------------------------------------------------------------------

def classify_document_type(text: str, max_length: int = 1500) -> str:
    """
    Classify the document type based on its textual content.

    Uses the Gemma LLM to intelligently identify the document category.
    Falls back to 'Other' on any error or unrecognised response.

    Args:
        text       : raw text of the page / document
        max_length : how many characters to feed to the LLM (truncation)

    Returns:
        One of the strings in VALID_DOC_TYPES.
    """
    text = text[:max_length]

    prompt = f"""
    Analyze this document and classify it into ONE of these categories:
    - Lender Fee Sheet: Lender Fee Sheet, loan estimate, Fee Details and Summary, loan amount, interest rate, ORIGINATION CHARGES, Annual Percentage Rate (APR), Total Interest Percentage (TIP), Calyx Form - LE1_1col.frm, Calyx Form - LE2_fixed.frm, Calyx Form - LE3_conf.frm, Calyx Form - feews.frm, monthly principal & interest, loan term, loan purpose, loan type (conventional/FHA/VA), rate lock, projected payments, Loan fees, lender charges, closing costs, loan terms, cash-to-close tables, FEES WORKSHEET
    - Mortgage: MORTGAGE, Security Instrument, FHA Wisconsin Mortgage, VA Mortgage, mortgagor, mortgagee, MERS, Mortgage Electronic Registration Systems, deed of trust, borrower owes lender, principal sum, recording data, parcel identifier number, FHA Case No, UNIFORM COVENANTS, BORROWER COVENANTS, Doc Yr, VMP, Wolters Kluwer
    - Discount Notice: CA Discount Notice, Notice of Available Discounts, fee reduction settlement program, disaster loans, churches or charitable non-profit organizations, employee rate, CTIC, TTCC, Ticor Title Company, Chicago Title Insurance Company, FNF Underwritten Title Company, FNF Underwriter, Section 2355.3, California Code of Regulations, credit for preliminary reports
    - Contract: Contract, Term of Employment, Probation, Legal agreement, SAMPLE CONTRACT OF EMPLOYMENT, Working Conditions, Interpretation of Agreement, Severability, service agreement, Compensation and Benefits, Termination of Employment, Annexure, Duties and Responsibilities, Confidentiality, Assignment
    - Preliminary Title Report: Preliminary Report, title report, vesting owners, CREDIT LINE / EQUITY LINE OF CREDIT CLOSURE REQUEST, STATEMENT OF INFORMATION, AFFIDAVIT, CERTIFICATION OF TRUST, CERTIFICATE OF ACKNOWLEDGEMENT, INFORMATIONAL NOTES SECTION, Ticor Title Company of California, CONFIDENTIAL INFORMATION STATEMENT, ORDER NO, LEGAL DESCRIPTION, TRANSMITTAL, wire instructions, ATTACHMENT ONE, APN, legal description, recorded deeds/deeds of trust, easements, recording numbers/dates, CLTA, ALTA
    - Pay Slip: Sample W-2, Pay slip, Payslip, 2020 W-2 and Earnings Summary, 1099, Salary statement, wage slip, earnings statement, Earnings, Amount, Employee Signature, Deductions, Payworks, Net Pay, Gross Earnings, Benefits & Accruals
    - Bank Statement: Bank Statement, TD Business Premier Checking, KE: CONTRACT LLC, DAILY ACCOUNT ACTIVITY, Electronic deposits, Account statement, transaction history, ACCOUNT SUMMARY, Beginning Balance, Ending Balance
    - Privacy Statement: Privacy Statement, Privacy Notice, Privacy Policy, information we collect, information we share, information we disclose, affiliated companies, nonaffiliated third parties, opt-out, privacy practices, Fidelity National Financial
    - Tax Document: Property Tax Document, Employee Reference Copy, tax return, tax form, Tax installments, fiscal year, code area, delinquency dates, penalties, exemptions
    - Insurance: Insurance policy, coverage document, LIMITATIONS ON COVERED RISKS, SCHEDULE B, PART I, RESIDENTIAL INSURANCE POLICY, COVERAGE RESIDENTIAL LOAN POLICY, STANDARD COVERAGE POLICY, EXCEPTIONS FROM COVERAGE, LAND ASSOCIATION LOAN POLICY, ENDORSEMENT-FORM 1, Policy conditions, exclusions, endorsements, coverage/limitations language, EXCLUSIONS FROM COVERAGE
    - Letter: Correspondence, memo, communication, from, to, Address of qualifying property, CA Settlement, Approximate date of transaction, To Whom It May Concern
    - ID Document: Driver's license, LIC. NO, ISS, EXP, HGT, WGT, passport, identification, KANSAS, DRIVER'S LICENSE
    - Resume: Resume, CV, professional profile, work history, Education, Employment History, Adult Care Experience, Career Summary, Functional Resume
    - Invoice: Line items, Item and Description, quantities, taxes, subtotals, grand total, Invoice#, Bill To, Ship To, Balance Due
    - Other: Doesn't fit other categories

    IMPORTANT RULES:
    - If the document contains "MORTGAGE" as a title and mentions "Security Instrument", "mortgagor", "MERS", or "FHA Case No", classify as "Mortgage" NOT "Contract" or "Lender Fee Sheet".
    - If the document mentions "Notice of Available Discounts", "CA Discount Notice", "fee reduction settlement program", or "disaster loans", classify as "Discount Notice" NOT "Other" or "Insurance".
    - If the document mentions "CREDIT LINE / EQUITY LINE OF CREDIT CLOSURE REQUEST", classify as "Preliminary Title Report".
    - If the document mentions "STATEMENT OF INFORMATION" or "TRANSMITTAL", classify as "Preliminary Title Report".

    Document sample:
    {text}

    Respond with ONLY the category name, nothing else.
    """

    try:
        # 32 tokens fits the longest label, "Preliminary Title Report".
        response = gemma_llm.complete(prompt, temperature=0, fast=True,
                                      thinking_budget=0, max_tokens=32)
        doc_type = response.text.strip()

        # Exact match (case-insensitive)
        for valid_type in VALID_DOC_TYPES:
            if doc_type.lower() == valid_type.lower():
                return valid_type

        # Fuzzy / partial match
        doc_type_lower = doc_type.lower()
        for valid_type in VALID_DOC_TYPES:
            if valid_type.lower() in doc_type_lower or doc_type_lower in valid_type.lower():
                return valid_type

        return "Other"

    except Exception as e:
        print(f"Classification error: {e}")
        return "Other"


# ---------------------------------------------------------------------------
# Document boundary detection
# ---------------------------------------------------------------------------

def detect_document_boundary(
    prev_text: str,
    curr_text: str,
    current_doc_type: str = None,
) -> bool:
    """
    Determine whether two consecutive PDF pages belong to the same document.

    Args:
        prev_text        : text from the previous page (tail ~500 chars)
        curr_text        : text from the current page (head ~500 chars)
        current_doc_type : known type of the current running document

    Returns:
        True  → same document (continue)
        False → new document starts on curr page
    """
    if not prev_text or not curr_text:
        return False

    # Slice texts to avoid sending entire pages unnecessarily
    prev_text_sliced = prev_text[-1500:]
    curr_text_sliced = curr_text[:1500]

    prompt = f"""
    Determine if these two pages are from the SAME document.

    Current document type: {current_doc_type or 'Unknown'}

    End of Previous Page:
    ...{prev_text_sliced}

    Start of Current Page:
    {curr_text_sliced}...

    Consider these factors:
    - Do section/paragraph numbers continue sequentially?
    - Is there a consistent document structure or format?
    - Does the content logically continue from the previous page?
    - Are headers, footers, or document titles the same?

    Answer ONLY 'Yes' if same document or 'No' if different document.
    """

    try:
        # Runs once per page, so it dominates ingest token cost.
        response = gemma_llm.complete(prompt, temperature=0, fast=True,
                                      thinking_budget=0, max_tokens=8)
        return response.text.strip().lower().startswith("yes")
    except Exception as e:
        print(f"Boundary detection error: {e}")
        # Default: keep pages together if uncertain
        return True
