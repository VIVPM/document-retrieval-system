"""document_classifier.py — LLM-based document type classification and
                          boundary detection between consecutive pages."""

from llm.llm_router import llm as gemma_llm
from logging_setup import get_logger

log = get_logger("drs.classifier")


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
        log.warning("classification returned nothing; using Other")
        return "Other"

    lowered = text.lower()
    for valid in VALID_DOC_TYPES:
        if lowered == valid.lower():
            return valid

    hits = [v for v in VALID_DOC_TYPES if v.lower() in lowered]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        log.warning("classification named several types; using Other",
                    extra={"reply": text[:120], "matched": hits})
        return "Other"

    log.warning("classification did not match any type; using Other",
                extra={"reply": text[:120]})
    return "Other"


def classify_document_type(text: str, max_length: int = 1500) -> str:
    """    Classify the document type based on its textual content."""
    text = text[:max_length]

    prompt = f"""
    This page comes from a mortgage loan file: the borrower's application plus
    the documents that support it. Classify the page by what the document IS,
    judged mainly by its title and layout, not by values it merely mentions.

    Categories:
    - Loan Application: the borrower's application (Uniform Residential Loan Application, URLA, Fannie Mae Form 1003, Freddie Mac Form 65). It COLLECTS the borrower's information: Borrower Information, Employment and Income, Assets and Liabilities, Loan and Property Information, Declarations, Military Service, Demographic Information, borrower signature.
    - Lender Fee Sheet: a lender's disclosure that STATES the loan's terms and costs: Loan Estimate, Closing Disclosure, fees worksheet, fee sheet. Loan Terms, Projected Payments, Origination Charges, Services You Can/Cannot Shop For, Prepaids, Initial Escrow, Cash to Close, APR, Total Interest Percentage (TIP), rate lock.
    - Pay Slip: a per-pay-period earnings statement from an employer: pay period, pay date, gross earnings, deductions, net pay, year-to-date columns, employee ID.
    - Bank Statement: a statement for a bank or credit union account over a period: account number, statement period, beginning/ending balance, deposits and withdrawals, daily activity or transaction list.
    - Tax Document: a tax form or tax record: W-2, 1099, tax return (Form 1040), tax transcript, property tax bill or statement (installments, delinquency dates, assessed value).
    - Preliminary Title Report: a title company's report on the property: Preliminary Report, order number, effective date, title vested in, legal description, APN, Schedule A / Schedule B exceptions (taxes, easements, recorded deeds of trust to be paid off), informational notes, CLTA/ALTA policies to be issued. Also the title company's related forms in the same package: Statement of Information, transmittal, payoff or credit line closure requests.
    - Mortgage: the recorded security instrument itself, titled "Mortgage" or "Deed of Trust": parties (borrower/mortgagor, lender/mortgagee, trustee, MERS), Uniform Covenants, Non-Uniform Covenants, signatures and notary acknowledgment.
    - Insurance: an insurance policy or its terms: homeowner's or hazard insurance declarations page, or a title insurance policy jacket with Covered Risks, Exclusions from Coverage, Conditions, endorsements.
    - Discount Notice: a title or escrow company's notice of available fee discounts (e.g. disaster-area, charitable organization, employee rates).
    - Privacy Statement: a company's privacy notice: what information it collects and shares, affiliates, opt-out choices.
    - Contract: an agreement between parties that is not a mortgage, e.g. purchase agreement, employment contract, service agreement.
    - Letter: correspondence: letter of explanation, gift letter, verification letter, memo addressed to someone.
    - ID Document: driver's license, passport, state ID.
    - Resume: a person's CV or work history.
    - Invoice: a bill with line items and a total due.
    - Other: none of the above.

    RULES for categories that are easy to confuse:
    - Loan Application vs Lender Fee Sheet: both show a loan amount and interest rate. If the page collects the borrower's details or answers (declarations, employment, assets), it is "Loan Application". If it states fees, payments and cash to close, it is "Lender Fee Sheet".
    - Pay Slip vs Tax Document: one pay period with net pay is "Pay Slip". A W-2 or 1099 covers a whole year and is "Tax Document".
    - Preliminary Title Report vs Mortgage: a title report LISTS a recorded deed of trust as an exception; that does not make it a "Mortgage". Only the instrument itself is a "Mortgage".
    - Preliminary Title Report vs Insurance: a preliminary report says what policy WILL be issued and lists Schedule B exceptions; it is not "Insurance". Only an issued policy is "Insurance".
    - Bank Statement vs Pay Slip: payroll deposits on an account statement are still a "Bank Statement".

    Document sample:
    {text}

    Respond with ONLY the category name, nothing else.
    """

    try:
        response = gemma_llm.complete(prompt, temperature=0, fast=True,
                                      thinking_budget=0, max_tokens=32)
        return _validate_doc_type(response.text or "")

    except Exception:
        log.exception("classification failed")
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

        log.warning("boundary detection unparseable; keeping pages together",
                    extra={"reply": (response.text or "")[:80]})
        return True
    except Exception:
        log.exception("boundary detection failed; keeping pages together")
        return True
