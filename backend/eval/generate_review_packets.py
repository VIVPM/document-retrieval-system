"""Generate six fictional PDF packets for automatic loan-file review evaluation.

Run: python backend/eval/generate_review_packets.py
The PDFs contain invented people, institutions, addresses, and account numbers.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf" / "review_eval_fictional"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    {"id": "01_clean", "borrower": "Mira Solen", "employer": "Northstar Lantern Works LLC",
     "app_income": 8000, "gross": 4000, "net": 3100, "app_balance": 18000,
     "bank_balance": 18000, "app_loan": 320000, "estimate_loan": 320000,
     "price": 400000, "third_party": None, "omit": None},
    {"id": "02_income_mismatch", "borrower": "Jonah Vale", "employer": "Blue Harbor Design LLC",
     "app_income": 10000, "gross": 4000, "net": 3100, "app_balance": 18000,
     "bank_balance": 18000, "app_loan": 320000, "estimate_loan": 320000,
     "price": 400000, "third_party": None, "omit": None},
    {"id": "03_balance_mismatch", "borrower": "Lina Oren", "employer": "Cedar Light Studio LLC",
     "app_income": 8000, "gross": 4000, "net": 3100, "app_balance": 30000,
     "bank_balance": 18000, "app_loan": 320000, "estimate_loan": 320000,
     "price": 400000, "third_party": None, "omit": None},
    {"id": "04_loan_amount_mismatch", "borrower": "Theo Arden", "employer": "Pine Bridge Analytics LLC",
     "app_income": 8000, "gross": 4000, "net": 3100, "app_balance": 18000,
     "bank_balance": 18000, "app_loan": 320000, "estimate_loan": 305000,
     "price": 400000, "third_party": None, "omit": None},
    {"id": "05_missing_pay_slip", "borrower": "Nina Corin", "employer": "Silver Maple Labs LLC",
     "app_income": 8000, "gross": 4000, "net": 3100, "app_balance": 18000,
     "bank_balance": 18000, "app_loan": 320000, "estimate_loan": 320000,
     "price": 400000, "third_party": None, "omit": "pay"},
    {"id": "06_missing_bank_statement", "borrower": "Omar Lysen", "employer": "Westbrook Craft LLC",
     "app_income": 8000, "gross": 4000, "net": 3100, "app_balance": 18000,
     "bank_balance": 18000, "app_loan": 320000, "estimate_loan": 320000,
     "price": 400000, "third_party": None, "omit": "bank"},
]


def money(value):
    return f"${value:,.2f}"


def page(pdf, title, subtitle, lines, case_id, page_number):
    w, h = 612, 792
    pdf.setFillColor(colors.HexColor("#142840"))
    pdf.rect(0, h - 92, w, 92, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 19)
    pdf.drawString(46, h - 48, title)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(46, h - 70, subtitle)

    pdf.setFillColor(colors.HexColor("#9B1C1C"))
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawCentredString(w / 2, h - 114, "SYNTHETIC TEST DOCUMENT - FICTIONAL DATA - NOT FOR LENDING")
    y = h - 150
    for label, value in lines:
        if label == "SECTION":
            y -= 13
            pdf.setFillColor(colors.HexColor("#20466B"))
            pdf.setFont("Helvetica-Bold", 11)
            pdf.drawString(46, y, value)
            y -= 19
            continue
        pdf.setFillColor(colors.HexColor("#3E4A59"))
        pdf.setFont("Helvetica", 10)
        pdf.drawString(46, y, label)
        pdf.setFillColor(colors.HexColor("#142840"))
        pdf.setFont("Helvetica-Bold", 10)
        wrapped = simpleSplit(str(value), "Helvetica-Bold", 10, 300)
        for j, part in enumerate(wrapped):
            pdf.drawString(256, y - 14 * j, part)
        y -= max(29, len(wrapped) * 14 + 10)
    pdf.setStrokeColor(colors.HexColor("#CBD5E1"))
    pdf.line(46, 48, w - 46, 48)
    pdf.setFillColor(colors.HexColor("#66788A"))
    pdf.setFont("Helvetica", 8)
    pdf.drawString(46, 34, f"Fictional review fixture {case_id}")
    pdf.drawRightString(w - 46, 34, f"Packet page {page_number}")
    pdf.showPage()


def build(case):
    path = OUT / f"{case['id']}.pdf"
    pdf = canvas.Canvas(str(path), pagesize=(612, 792), pageCompression=1)
    pdf.setTitle(f"Fictional loan review packet {case['id']}")
    n = 1
    person = case["borrower"]
    employer = case["employer"]
    page(pdf, "Loan Application", f"Borrower claims - submitted 10/01/2026",
         [("Borrower Name:", person), ("Employer or Business Name:", employer),
          ("Start Date:", "01/15/2022"),
          ("TOTAL Gross Monthly Income:", money(case["app_income"])),
          ("Checking Account Balance:", money(case["app_balance"])),
          ("Loan Amount Requested:", money(case["app_loan"])),
          ("Purchase Price:", money(case["price"])),
          ("Undisclosed Funds From Another Party?:", "NO"),
          ("Co-Borrower:", "None"),
          ("SECTION", "Applicant declaration"),
          ("Other monthly debts:", "None disclosed")], case["id"], n)
    n += 1
    if case["omit"] != "pay":
        page(pdf, "Pay Slip", f"{employer} - employee {person}",
             [("Company:", employer), ("Employee Name:", person),
              ("Pay Frequency:", "Semi-monthly"),
              ("Period Start:", "09/16/2026"), ("Period End:", "09/30/2026"),
              ("Gross Pay:", money(case["gross"])),
              ("Net Pay:", money(case["net"])),
              ("YTD Gross:", money(case["gross"] * 18))], case["id"], n)
        n += 1
    if case["omit"] != "bank":
        page(pdf, "Bank Statement", f"Fictional Community Bank - checking account 0099",
             [("Account Holder:", person), ("Statement Period:", "09/01/2026 - 09/30/2026"),
              ("Beginning Balance:", money(case["bank_balance"] - 6200)),
              ("Ending Balance:", money(case["bank_balance"])),
              ("SECTION", "Transactions"),
              ("09/15/2026", f"{employer} PAYROLL DEPOSIT +{money(case['net'])}"),
              ("09/30/2026", f"{employer} PAYROLL DEPOSIT +{money(case['net'])}")],
             case["id"], n)
        n += 1
    page(pdf, "Loan Estimate", f"Fictional Home Lender - issued 10/02/2026",
         [("Borrower:", person), ("Loan Amount:", money(case["estimate_loan"])),
          ("Sale Price:", money(case["price"])),
          ("Interest Rate:", "6.25%"),
          ("Monthly Principal & Interest:", "$1,970.00"),
          ("Estimated Closing Costs:", "$7,800.00")], case["id"], n)
    n += 1
    page(pdf, "Preliminary Title Report", "Property: 41 Lantern Way, Fableton, OR 97000",
         [("Title Vested In:", "Fictional Seller LLC"),
          ("Report Date:", "10/02/2026"),
          ("SECTION", "Exceptions and liens"),
          ("Recorded Liens:", "None"),
          ("Recorded Easements:", "None")], case["id"], n)
    pdf.save()
    print(f"{path} ({n} pages)")


if __name__ == "__main__":
    for case in CASES:
        build(case)
