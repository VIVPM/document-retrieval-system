"""Reducto's table and page output must fit the existing extractor contract."""

import unittest

from core.reducto_adapter import pages_from_reducto


class ReductoAdapterTest(unittest.TestCase):
    def test_table_rows_become_searchable_text_and_form_pairs(self):
        response = {"usage": {"num_pages": 2}, "result": {"chunks": [{"blocks": [
            {"type": "Table", "bbox": {"page": 2, "top": 0.3, "left": 0.1},
             "content": "<table><tr><td><b>Loan Amount</b></td><td>$380,000.00</td></tr>"
                        "<tr><td>Purchase Price</td><td>$475,000.00</td></tr></table>"}
        ]}]}}
        pages = pages_from_reducto(response)
        self.assertEqual(list(pages), ["1", "2"])
        self.assertEqual(pages["1"], [])
        self.assertEqual(pages["2"][0]["content"],
                         "Loan Amount | $380,000.00\nPurchase Price | $475,000.00")
        self.assertEqual([(b["key"], b["value"]) for b in pages["2"][1:]],
                         [("Loan Amount", "$380,000.00"), ("Purchase Price", "$475,000.00")])


if __name__ == "__main__":
    unittest.main()
