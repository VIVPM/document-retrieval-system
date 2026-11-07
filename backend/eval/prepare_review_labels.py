"""Prepare proposed field labels for the six fictional review packets.

These labels are derived from fixture specifications and have NOT been reviewed
by a human. They must not be reported as human ground truth or human-labeled
accuracy until a reviewer approves or corrects them.

Run: python backend/eval/prepare_review_labels.py
"""

import csv
import hashlib
import json

from generate_review_packets import CASES, OUT, money


LABELS = OUT / "proposed_review_labels.csv"
MANIFEST = OUT / "proposed_review_manifest.json"
FIELDS = [
    "label_status", "case_id", "packet_file", "packet_sha256", "check_key",
    "field_key", "expected_document_type", "expected_field_value", "source_page",
    "expected_discrepancy_class", "expected_review_status", "missing_evidence_outcome", "reviewer_decision",
    "reviewer_note",
]


def row(case, digest, check, field, doc_type, value, page, outcome, missing="none"):
    return {
        "label_status": "proposed_unreviewed",
        "case_id": case["id"],
        "packet_file": f"{case['id']}.pdf",
        "packet_sha256": digest,
        "check_key": check,
        "field_key": field,
        "expected_document_type": doc_type,
        "expected_field_value": value,
        "source_page": page,
        "expected_discrepancy_class": outcome,
        "expected_review_status": "missing_document" if outcome == "evidence_missing" else outcome,
        "missing_evidence_outcome": missing,
        "reviewer_decision": "",
        "reviewer_note": "",
    }


def case_rows(case, digest):
    missing_pay = case["omit"] == "pay"
    missing_bank = case["omit"] == "bank"
    pay_page = "" if missing_pay else 2
    bank_page = "" if missing_bank else (2 if missing_pay else 3)
    estimate_page = 3 if missing_pay or missing_bank else 4

    income = "evidence_missing" if missing_pay else (
        "mismatch" if case["app_income"] != case["gross"] * 2 else "match")
    employer = "evidence_missing" if missing_pay else "match"
    balance = "evidence_missing" if missing_bank else (
        "mismatch" if case["app_balance"] != case["bank_balance"] else "match")
    loan = "mismatch" if case["app_loan"] != case["estimate_loan"] else "match"
    price = "match"
    no_pay = "pay_slip_absent; report missing document; do not claim a mismatch"
    no_bank = "bank_statement_absent; report missing document; do not claim a mismatch"

    return [
        row(case, digest, "income", "stated_monthly_income", "Loan Application",
            money(case["app_income"]), 1, income, no_pay if missing_pay else "none"),
        row(case, digest, "income", "gross_pay", "Pay Slip",
            "" if missing_pay else money(case["gross"]), pay_page, income,
            no_pay if missing_pay else "none"),
        row(case, digest, "employer", "employer", "Loan Application",
            case["employer"], 1, employer, no_pay if missing_pay else "none"),
        row(case, digest, "employer", "employer", "Pay Slip",
            "" if missing_pay else case["employer"], pay_page, employer,
            no_pay if missing_pay else "none"),
        row(case, digest, "balance", "checking_balance", "Loan Application",
            money(case["app_balance"]), 1, balance, no_bank if missing_bank else "none"),
        row(case, digest, "balance", "ending_balance", "Bank Statement",
            "" if missing_bank else money(case["bank_balance"]), bank_page, balance,
            no_bank if missing_bank else "none"),
        row(case, digest, "loan_amount", "loan_amount", "Loan Application",
            money(case["app_loan"]), 1, loan),
        row(case, digest, "loan_amount", "loan_amount", "Loan Estimate",
            money(case["estimate_loan"]), estimate_page, loan),
        row(case, digest, "price", "purchase_price", "Loan Application",
            money(case["price"]), 1, price),
        row(case, digest, "price", "sale_price", "Loan Estimate",
            money(case["price"]), estimate_page, price),
    ]


def main():
    rows = []
    packets = []
    for case in CASES:
        packet = OUT / f"{case['id']}.pdf"
        digest = hashlib.sha256(packet.read_bytes()).hexdigest()
        packets.append({"case_id": case["id"], "file": packet.name, "sha256": digest})
        rows.extend(case_rows(case, digest))
    assert len(rows) == 60

    with LABELS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "proposed_unreviewed",
        "provenance": "Generated deterministically from fictional packet specifications; not human labels or LLM ground truth.",
        "field_rows": len(rows),
        "comparison_checks": len(CASES) * 5,
        "packets": packets,
        "labels_file": LABELS.name,
        "labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(rows)} proposed field rows across {len(packets)} packets")
    print(LABELS)
    print(MANIFEST)


if __name__ == "__main__":
    main()
