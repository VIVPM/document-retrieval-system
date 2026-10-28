"""Regressions for deposit explanations and Loan Estimate citations."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.document_classifier_jev import classify_document_type
from core.review import SCHEMAS, fields_from_forms, fields_from_llm, run_rules


def field(value, raw, page):
    return {"value": value, "raw": raw, "page": page, "source": "form"}


def documents(gross=4375):
    return [
        {"doc_type": "Loan Application", "fields": {
            "stated_monthly_income": field(9500, "$9,500.00", 1),
            "undisclosed_funds": field(False, "NO", 2),
            "loan_amount": field(420000, "$420,000.00", 2)}, "lists": {}},
        {"doc_type": "Pay Slip", "fields": {
            "gross_pay": field(gross, f"${gross:,.2f}", 4),
            "pay_frequency": field("Semi-monthly", "Semi-monthly", 4)}, "lists": {}},
        {"doc_type": "Bank Statement", "fields": {}, "lists": {"deposits": [
            {"date": "07/12", "description": "TRANSFER IN FROM ACCT ***9902 R WHITFIELD",
             "amount": "+$12,000.00", "amount_value": 12000, "page": 5}]}},
        {"doc_type": "Loan Estimate", "fields": {
            "loan_amount": field(420000, "$420,000.00", 6)}, "lists": {}},
    ]


class WalkthroughAccuracyTest(unittest.TestCase):
    def test_explicit_estimate_headings_skip_classifier(self):
        with patch("core.document_classifier_jev._ask") as ask:
            for heading in ("Loan Estimate", "LOAN ESTIMATE", "Loan Estimate — Closing Cost Details"):
                with self.subTest(heading=heading):
                    self.assertEqual(classify_document_type(heading + "\nLoan ID # 123"), "Loan Estimate")
            ask.assert_not_called()

    def test_incidental_reference_does_not_relabel_fee_sheet(self):
        with patch("core.document_classifier_jev._ask", return_value={"choice": "Lender Fee Sheet"}):
            self.assertEqual(classify_document_type("Fees Worksheet\nCompare with your Loan Estimate"), "Lender Fee Sheet")

    def test_new_and_legacy_types_both_participate_in_review(self):
        for doc_type in ("Loan Estimate", "Lender Fee Sheet"):
            docs = documents()
            docs[-1]["doc_type"] = doc_type
            checks = run_rules(docs)
            check = next(c for c in checks if c["id"] == "loan_amount")
            self.assertEqual(check["status"], "match")
            self.assertEqual(check["evidence"][1]["doc_type"], doc_type)
            self.assertEqual(fields_from_forms(doc_type, [("Loan Amount", "$420,000.00", 5)])["loan_amount"]["page"], 6)

    def test_declaration_citation_matches_question_and_answer_not_loan_number(self):
        doc = SimpleNamespace(page_start=0, text="", blocks=[
            SimpleNamespace(page_num=0, content="Lender Loan No. 123\nIndividual credit: no co-borrower"),
            SimpleNamespace(page_num=1, content="C. Are you borrowing any money for this real estate transaction "
                            "or obtaining any money from another party that you have not disclosed on this loan application? | NO")])
        spec = {"undisclosed_funds": SCHEMAS["Loan Application"]["fields"]["undisclosed_funds"]}
        fields, _, dropped = fields_from_llm("Loan Application", doc, spec,
            complete=lambda _: '{"fields":{"undisclosed_funds":"NO"}}')
        self.assertEqual(fields["undisclosed_funds"]["page"], 2)
        self.assertEqual(dropped, 0)
        fields, _, dropped = fields_from_llm("Loan Application", doc, spec,
            complete=lambda _: '{"fields":{"undisclosed_funds":"YES"}}')
        self.assertNotIn("undisclosed_funds", fields)
        self.assertEqual(dropped, 1)
        doc.blocks = doc.blocks[:1]
        fields, _, dropped = fields_from_llm("Loan Application", doc, spec,
            complete=lambda _: '{"fields":{"undisclosed_funds":"NO"}}')
        self.assertNotIn("undisclosed_funds", fields)
        self.assertEqual(dropped, 1)

    def test_sidebar_threshold_and_evidence(self):
        for gross, threshold in ((4375, "$2,187.50"), (2000, "$1,000.00")):
            checks = run_rules(documents(gross))
            check = next(c for c in checks if c["id"] == "large_deposits")
            self.assertEqual(check["status"], "review")
            self.assertIn(threshold, check["detail"])
            self.assertIn("does not prove third-party funds or a contradiction", check["detail"])
            self.assertTrue(any(e["page"] == 2 and e["value"] == "NO" for e in check["evidence"]))
            self.assertTrue(any(e["page"] == 4 and e["value"] == "Semi-monthly" for e in check["evidence"]))


if __name__ == "__main__":
    unittest.main()
