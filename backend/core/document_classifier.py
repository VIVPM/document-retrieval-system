"""document_classifier.py — LLM-based document type classification and
                          boundary detection between consecutive pages."""

from llm.llm_router import llm as gemma_llm


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
    "Loan Application",
    "Preliminary Title Report",
    "Invoice",
    "Other",
]


def _validate_doc_type(raw: str) -> str:
    """Coerce a model reply to one of VALID_DOC_TYPES, or "Other"."""
    text = (raw or "").strip()
    if not text:
        print("Classification returned nothing; using Other")
        return "Other"

    lowered = text.lower()
    for valid in VALID_DOC_TYPES:
        if lowered == valid.lower():
            return valid

    hits = [v for v in VALID_DOC_TYPES if v.lower() in lowered]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f"Classification named several types {hits}; using Other")
        return "Other"

    print(f"Classification did not match any type ({text[:60]!r}); using Other")
    return "Other"


def classify_document_type(text: str, max_length: int = 1500) -> str:
    """    Classify the document type based on its textual content."""
    text = text[:max_length]

    prompt = f"""
    Analyze this document and classify it into ONE of these categories:
    - Lender Fee Sheet: Lender Fee Sheet, loan estimate, Fee Details and Summary, loan amount, interest rate, ORIGINATION CHARGES, Annual Percentage Rate (APR), Total Interest Percentage (TIP), Calyx Form - LE1_1col.frm, Calyx Form - LE2_fixed.frm, Calyx Form - LE3_conf.frm, Calyx Form - feews.frm, monthly principal & interest, loan term, loan purpose, loan type (conventional/FHA/VA), rate lock, projected payments, Loan fees, lender charges, closing costs, loan terms, cash-to-close tables, FEES WORKSHEET
    - Loan Application: Uniform Residential Loan Application, URLA, Form 1003, Fannie Mae Form 1003, Freddie Mac Form 65, loan application form, home loan application, applicant details, co-applicant, co-borrower, Borrower Information, Personal Information, Employment and Income, current employer, years on the job, gross monthly income, Assets and Liabilities, existing loans / EMIs, Declarations, Demographic Information, Military Service, Acknowledgments and Agreements, loan amount requested, applicant's signature
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
    - If the document asks the applicant to PROVIDE personal, employment, income, asset or liability details (e.g. "Borrower Information", "Employment and Income", "Declarations", Form 1003 / URLA), classify as "Loan Application" NOT "Lender Fee Sheet", even though both mention loan amount and interest rate. A Lender Fee Sheet STATES fees and costs; a Loan Application COLLECTS the applicant's information.
    - If the document contains "MORTGAGE" as a title and mentions "Security Instrument", "mortgagor", "MERS", or "FHA Case No", classify as "Mortgage" NOT "Contract" or "Lender Fee Sheet".
    - If the document mentions "Notice of Available Discounts", "CA Discount Notice", "fee reduction settlement program", or "disaster loans", classify as "Discount Notice" NOT "Other" or "Insurance".
    - If the document mentions "CREDIT LINE / EQUITY LINE OF CREDIT CLOSURE REQUEST", classify as "Preliminary Title Report".
    - If the document mentions "STATEMENT OF INFORMATION" or "TRANSMITTAL", classify as "Preliminary Title Report".

    Document sample:
    {text}

    Respond with ONLY the category name, nothing else.
    """

    try:
        response = gemma_llm.complete(prompt, temperature=0, fast=True,
                                      thinking_budget=0, max_tokens=32)
        return _validate_doc_type(response.text or "")

    except Exception as e:
        print(f"Classification error: {e}")
        return "Other"


def detect_document_boundary(
    prev_text: str,
    curr_text: str,
    current_doc_type: str = None,
) -> bool:
    """    Determine whether two consecutive PDF pages belong to the same document."""
    if not prev_text or not curr_text:
        return False

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
        response = gemma_llm.complete(prompt, temperature=0, fast=True,
                                      thinking_budget=0, max_tokens=8)
        answer = (response.text or "").strip().lower()
        if answer.startswith("yes"):
            return True
        if answer.startswith("no"):
            return False

        print(f"Boundary detection unparseable ({(response.text or '')[:40]!r}); "
              "keeping pages together")
        return True
    except Exception as e:
        print(f"Boundary detection error: {e}")
        return True
