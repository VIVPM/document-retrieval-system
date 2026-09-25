"""Generate one SYNTHETIC borrower's loan packet, plus an answer key."""

import os

import fitz

OUT = os.path.dirname(os.path.abspath(__file__))

B = dict(
    name="Daniel R. Whitfield", dob="03/14/1988", ssn="XXX-XX-4417",
    phone="(916) 555-0147", email="d.whitfield@example.com",
    marital="Unmarried", dependents="0",
    cur_addr="2217 Maple Grove Ln, Apt 4B, Sacramento, CA 95818",
    cur_housing="Rent", cur_rent=2150.00, cur_years="3 years 2 months",
)
EMP = dict(
    name="Sierra Ridge Logistics, Inc.", addr="900 Riverfront Pkwy, Sacramento, CA 95814",
    phone="(916) 555-0190", title="Senior Operations Analyst", start="06/01/2019",
    years_line="9", emp_id="SRL-20417",
)
STATED_MONTHLY = 9500.00
SEMI_GROSS = 4375.00
PROP = dict(
    addr="4815 Harbor View Dr, Sacramento, CA 95831", apn="031-0482-017-0000",
    price=525000.00, occupancy="Primary Residence", units="1",
)
LOAN = dict(amount=420000.00, rate=6.375, term_years=30, purpose="Purchase",
            product="Fixed Rate", type="Conventional", lock_until="09/15/2026",
            lender="Pinecrest Home Lending", nmls="1840226", deposit=10000.00)
BANK = dict(name="Golden State Credit Union", acct="XXXXXX4471",
            period="07/01/2026 - 07/31/2026", begin=38412.56)
AUTO = dict(creditor="Capitol Auto Finance", balance=14200.00, monthly=486.00)
CARD = dict(creditor="Chase Card Services", balance=3100.00, monthly=85.00)
TITLE = dict(company="Cascade Title Company", order="2026-04471-SR",
             officer="Maria Delgado", officer_phone="(916) 555-0122",
             effective="07/20/2026",
             vested="Thomas J. Albrecht and Linda M. Albrecht, husband and wife, as joint tenants")

money = lambda x: f"${x:,.2f}"
r = LOAN["rate"] / 100 / 12
n = LOAN["term_years"] * 12
PI = round(LOAN["amount"] * r / (1 - (1 + r) ** -n), 2)
DOWN = PROP["price"] - LOAN["amount"]
LTV = LOAN["amount"] / PROP["price"] * 100
TAX_MO = round(PROP["price"] * 0.011 / 12, 2)
INS_MO = 125.00
ESCROW_MO = TAX_MO + INS_MO

deductions = [("Federal Income Tax", 612.50), ("CA State Income Tax", 218.75),
              ("Social Security", round(SEMI_GROSS * 0.062, 2)),
              ("Medicare", round(SEMI_GROSS * 0.0145, 2)),
              ("401(k) 5%", round(SEMI_GROSS * 0.05, 2)), ("Medical / Dental", 145.00)]
DED = round(sum(v for _, v in deductions), 2)
NET = round(SEMI_GROSS - DED, 2)
PAYSLIPS = [("07/01/2026 - 07/15/2026", "07/15/2026", 13),
            ("07/16/2026 - 07/31/2026", "07/31/2026", 14)]

txns = [("07/01", "ACH DEBIT  MAPLE GROVE APARTMENTS RENT", -2150.00),
        ("07/03", "ACH DEBIT  PG&E UTILITY PAYMENT", -142.87),
        ("07/08", "ACH DEBIT  CAPITOL AUTO FINANCE", -AUTO["monthly"]),
        ("07/12", "TRANSFER IN  FROM ACCT ***9902  R WHITFIELD", 12000.00),
        ("07/15", "DIRECT DEP  SIERRA RIDGE LOGISTICS PAYROLL", NET),
        ("07/18", "ONLINE PMT  CHASE CARD SERVICES", -620.00),
        ("07/22", "POS DEBIT  RALEY'S #118 SACRAMENTO", -214.33),
        ("07/31", "DIRECT DEP  SIERRA RIDGE LOGISTICS PAYROLL", NET)]
bal, rows = BANK["begin"], []
for d, desc, amt in txns:
    bal = round(bal + amt, 2)
    rows.append((d, desc, amt, bal))
END_BAL = bal
DEPOSITS = round(sum(a for *_, a in txns if a > 0), 2)
WITHDRAWALS = round(-sum(a for *_, a in txns if a < 0), 2)

A = [("Origination Fee (0.25% of loan amount)", round(LOAN["amount"] * 0.0025, 2)),
     ("Application Fee", 395.00), ("Underwriting Fee", 895.00)]
Bs = [("Appraisal Fee", 650.00), ("Credit Report Fee", 45.00),
      ("Flood Determination Fee", 12.00), ("Tax Service Fee", 85.00)]
C = [("Title - Lender's Title Policy", 1480.00), ("Title - Settlement Agent Fee", 1150.00),
     ("Title - Title Search", 325.00)]
E = [("Recording Fees", 175.00),
     ("County Transfer Tax", round(PROP["price"] / 1000 * 1.10, 2))]
PREPAID_INT = round(LOAN["amount"] * LOAN["rate"] / 100 / 365 * 15, 2)
F = [("Homeowner's Insurance Premium (12 months)", INS_MO * 12),
     ("Prepaid Interest (15 days)", PREPAID_INT)]
G = [("Homeowner's Insurance (2 months)", INS_MO * 2),
     ("Property Taxes (3 months)", round(TAX_MO * 3, 2))]
tot = lambda xs: round(sum(v for _, v in xs), 2)
D_TOTAL = round(tot(A) + tot(Bs) + tot(C), 2)
I_TOTAL = round(tot(E) + tot(F) + tot(G), 2)
J_TOTAL = round(D_TOTAL + I_TOTAL, 2)
CASH_TO_CLOSE = round(DOWN + J_TOTAL - LOAN["deposit"], 2)
APR = "6.498%"

CSS = """
* { font-family: sans-serif; }
body { font-size: 9.5pt; color: #111; }
h1 { font-size: 15pt; margin: 0 0 2pt 0; }
h2 { font-size: 11pt; margin: 9pt 0 3pt 0; background: #e8ecf4; padding: 2pt 4pt; }
p { margin: 2pt 0; }
table { width: 100%; border-collapse: collapse; margin: 2pt 0 4pt 0; }
td, th { border: 0.6pt solid #999; padding: 2pt 4pt; vertical-align: top; }
th { background: #f2f2f2; text-align: left; }
.r { text-align: right; }
.small { font-size: 7.5pt; color: #555; }
.banner { font-size: 7pt; color: #a00; text-align: center; }
"""
BANNER = '<p class="banner">SYNTHETIC TEST DATA — all names, numbers and addresses are fictitious.</p>'


def pdf(filename, pages):
    doc = fitz.open()
    for html in pages:
        page = doc.new_page(width=612, height=792)
        rc = page.insert_htmlbox(fitz.Rect(40, 36, 572, 756), BANNER + html, css=CSS)
        if rc[0] < 0:
            raise RuntimeError(f"{filename}: content overflows the page")
    doc.save(os.path.join(OUT, filename), garbage=3, deflate=True)
    return doc.page_count


def kv(rows):
    return "<table>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"


def items(rows, total_label):
    body = "".join(f"<tr><td>{k}</td><td class='r'>{money(v)}</td></tr>" for k, v in rows)
    return (f"<table>{body}<tr><th>{total_label}</th>"
            f"<th class='r'>{money(tot(rows))}</th></tr></table>")


app1 = f"""
<h1>Uniform Residential Loan Application</h1>
<p class="small">Freddie Mac Form 65 • Fannie Mae Form 1003 &nbsp;|&nbsp; Lender: {LOAN['lender']}
 &nbsp;|&nbsp; Lender Loan No. PHL-2026-38815</p>
<h2>Section 1: Borrower Information</h2>
<p><b>1a. Personal Information</b></p>
{kv([("Name", B['name']), ("Social Security Number", B['ssn']), ("Date of Birth", B['dob']),
     ("Marital Status", B['marital']), ("Dependents", B['dependents']),
     ("Phone / Email", f"{B['phone']} / {B['email']}"),
     ("Current Address", B['cur_addr']),
     ("How long at current address", B['cur_years']),
     ("Housing", f"{B['cur_housing']} ({money(B['cur_rent'])}/month)"),
     ("Type of Credit", "I am applying for individual credit (no co-borrower)")])}
<p><b>1b. Current Employment / Self-Employment and Income</b></p>
{kv([("Employer or Business Name", EMP['name']), ("Employer Address", EMP['addr']),
     ("Employer Phone", EMP['phone']), ("Position or Title", EMP['title']),
     ("Start Date", EMP['start']), ("Years in this line of work", EMP['years_line']),
     ("Gross Monthly Income — Base", money(STATED_MONTHLY)),
     ("Overtime / Bonus / Commission", "$0.00"),
     ("TOTAL Gross Monthly Income", money(STATED_MONTHLY))])}
<p><b>1c.–1e. Additional / Previous Employment, Other Income:</b> Does not apply.</p>
<h2>Section 2: Financial Information — Assets and Liabilities</h2>
<p><b>2a. Assets — Bank, Retirement, and Other Accounts</b></p>
<table><tr><th>Account Type</th><th>Financial Institution</th><th>Account Number</th><th class='r'>Cash or Market Value</th></tr>
<tr><td>Checking</td><td>{BANK['name']}</td><td>{BANK['acct']}</td><td class='r'>{money(52400)}</td></tr>
<tr><td>Retirement (401k)</td><td>Fidelity Investments</td><td>XXXX8830</td><td class='r'>{money(61300)}</td></tr></table>
<p><b>2c. Liabilities — Credit Cards, Other Debts, and Leases</b></p>
<table><tr><th>Account Type</th><th>Company Name</th><th class='r'>Unpaid Balance</th><th class='r'>Monthly Payment</th></tr>
<tr><td>Installment (Auto)</td><td>{AUTO['creditor']}</td><td class='r'>{money(AUTO['balance'])}</td><td class='r'>{money(AUTO['monthly'])}</td></tr>
<tr><td>Revolving</td><td>{CARD['creditor']}</td><td class='r'>{money(CARD['balance'])}</td><td class='r'>{money(CARD['monthly'])}</td></tr></table>
"""
app2 = f"""
<h1>Uniform Residential Loan Application (continued)</h1>
<p class="small">Borrower: {B['name']} &nbsp;|&nbsp; Lender Loan No. PHL-2026-38815</p>
<h2>Section 4: Loan and Property Information</h2>
{kv([("Loan Amount", money(LOAN['amount'])), ("Loan Purpose", LOAN['purpose']),
     ("Property Address", PROP['addr']), ("Number of Units", PROP['units']),
     ("Property Value / Purchase Price", money(PROP['price'])),
     ("Occupancy", PROP['occupancy']),
     ("Gifts or Grants you have been given or will receive", "None")])}
<h2>Section 5: Declarations</h2>
<table><tr><th>Question</th><th>Answer</th></tr>
<tr><td>A. Will you occupy the property as your primary residence?</td><td>YES</td></tr>
<tr><td>B. Have you had an ownership interest in another property in the last three years?</td><td>NO</td></tr>
<tr><td>C. Are you borrowing any money for this real estate transaction (e.g., money for your closing costs or down payment) or obtaining any money from another party that you have not disclosed on this loan application?</td><td>NO</td></tr>
<tr><td>D. Have you applied for any new credit not disclosed on this application?</td><td>NO</td></tr>
<tr><td>E. Are there outstanding judgments against you?</td><td>NO</td></tr>
<tr><td>F. Have you declared bankruptcy within the past 7 years?</td><td>NO</td></tr></table>
<h2>Section 6: Acknowledgments and Agreements</h2>
<p>I certify that the information provided in this application is true and correct as of the date set forth
opposite my signature. I understand that any intentional or negligent misrepresentation may result in civil
liability and/or criminal penalties.</p>
<h2>Section 7: Military Service</h2>
<p>Did you (or your deceased spouse) ever serve in the United States Armed Forces? NO</p>
<h2>Section 8: Demographic Information</h2>
<p>I do not wish to provide this information.</p>
<p style="margin-top:14pt"><b>Borrower Signature:</b> /s/ {B['name']} &nbsp;&nbsp;&nbsp; <b>Date:</b> 07/24/2026</p>
<p><b>Loan Originator:</b> Karen Liu, NMLS ID 2291054 &nbsp;|&nbsp; {LOAN['lender']}, NMLS ID {LOAN['nmls']}</p>
"""


def payslip(period, paydate, n_periods):
    ytd_g, ytd_n = SEMI_GROSS * n_periods, NET * n_periods
    ded_rows = "".join(f"<tr><td>{k}</td><td class='r'>{money(v)}</td><td class='r'>{money(v * n_periods)}</td></tr>"
                       for k, v in deductions)
    return f"""
<h1>{EMP['name']}</h1>
<p>{EMP['addr']} &nbsp;|&nbsp; {EMP['phone']}</p>
<h2>EARNINGS STATEMENT — Pay Slip</h2>
{kv([("Employee Name", B['name']), ("Employee ID", EMP['emp_id']), ("Position", EMP['title']),
     ("Pay Period", period), ("Pay Date", paydate), ("Pay Frequency", "Semi-monthly"),
     ("Annual Base Salary", money(SEMI_GROSS * 24))])}
<table><tr><th>Earnings</th><th class='r'>Hours</th><th class='r'>Current</th><th class='r'>Year to Date</th></tr>
<tr><td>Regular Salary</td><td class='r'>86.67</td><td class='r'>{money(SEMI_GROSS)}</td><td class='r'>{money(ytd_g)}</td></tr>
<tr><th>Gross Earnings</th><th></th><th class='r'>{money(SEMI_GROSS)}</th><th class='r'>{money(ytd_g)}</th></tr></table>
<table><tr><th>Deductions</th><th class='r'>Current</th><th class='r'>Year to Date</th></tr>{ded_rows}
<tr><th>Total Deductions</th><th class='r'>{money(DED)}</th><th class='r'>{money(DED * n_periods)}</th></tr></table>
<table><tr><th>NET PAY</th><th class='r'>{money(NET)}</th><th class='r'>YTD Net {money(ytd_n)}</th></tr></table>
<p>Deposited to: {BANK['name']} checking {BANK['acct']}</p>
"""


bank = f"""
<h1>{BANK['name']}</h1>
<p>PO Box 15410, Sacramento, CA 95851 &nbsp;|&nbsp; Member Services (916) 555-0100</p>
<h2>Account Statement — Everyday Checking</h2>
{kv([("Member", B['name']), ("Mailing Address", B['cur_addr']),
     ("Account Number", BANK['acct']), ("Statement Period", BANK['period'])])}
<h2>ACCOUNT SUMMARY</h2>
{kv([("Beginning Balance (07/01/2026)", money(BANK['begin'])),
     ("Total Deposits and Credits", money(DEPOSITS)),
     ("Total Withdrawals and Debits", money(WITHDRAWALS)),
     ("Ending Balance (07/31/2026)", money(END_BAL))])}
<h2>DAILY ACCOUNT ACTIVITY</h2>
<table><tr><th>Date</th><th>Description</th><th class='r'>Amount</th><th class='r'>Balance</th></tr>
{''.join(f"<tr><td>{d}</td><td>{desc}</td><td class='r'>{'' if a < 0 else '+'}{money(a).replace('$-', '-$')}</td><td class='r'>{money(b)}</td></tr>" for d, desc, a, b in rows)}
</table>
<p class="small">Dividends earned this period: $0.00. This statement is provided for your records.</p>
"""

le1 = f"""
<h1>Loan Estimate</h1>
<p>{LOAN['lender']} &nbsp;|&nbsp; 2400 Capitol Mall, Suite 600, Sacramento, CA 95816 &nbsp;|&nbsp; NMLS ID {LOAN['nmls']}</p>
{kv([("Date Issued", "07/26/2026"), ("Applicant", B['name']), ("Property", PROP['addr']),
     ("Sale Price", money(PROP['price'])), ("Loan Term", f"{LOAN['term_years']} years"),
     ("Purpose", LOAN['purpose']), ("Product", LOAN['product']), ("Loan Type", LOAN['type']),
     ("Loan ID #", "PHL-2026-38815"),
     ("Rate Lock", f"YES, your interest rate is locked until {LOAN['lock_until']} at 5:00 p.m. PT")])}
<h2>Loan Terms</h2>
{kv([("Loan Amount", money(LOAN['amount'])), ("Interest Rate", f"{LOAN['rate']:.3f}%"),
     ("Monthly Principal &amp; Interest", money(PI)),
     ("Prepayment Penalty", "NO"), ("Balloon Payment", "NO")])}
<h2>Projected Payments (Years 1–30)</h2>
{kv([("Principal &amp; Interest", money(PI)),
     ("Estimated Escrow (property taxes + homeowner's insurance)", money(ESCROW_MO)),
     ("Estimated Total Monthly Payment", money(PI + ESCROW_MO)),
     ("Estimated Taxes, Insurance &amp; Assessments", f"{money(ESCROW_MO)} a month — In escrow: YES")])}
<h2>Costs at Closing</h2>
{kv([("Estimated Closing Costs", money(J_TOTAL)), ("Estimated Cash to Close", money(CASH_TO_CLOSE))])}
"""
le2 = f"""
<h1>Loan Estimate — Closing Cost Details</h1>
<p class="small">Loan ID # PHL-2026-38815 &nbsp;|&nbsp; Applicant: {B['name']}</p>
<h2>Loan Costs</h2>
<p><b>A. Origination Charges</b></p>{items(A, "Total Origination Charges")}
<p><b>B. Services You Cannot Shop For</b></p>{items(Bs, "Total")}
<p><b>C. Services You Can Shop For</b></p>{items(C, "Total")}
{kv([("D. TOTAL LOAN COSTS (A + B + C)", money(D_TOTAL))])}
<h2>Other Costs</h2>
<p><b>E. Taxes and Other Government Fees</b></p>{items(E, "Total")}
<p><b>F. Prepaids</b></p>{items(F, "Total")}
<p><b>G. Initial Escrow Payment at Closing</b></p>{items(G, "Total")}
{kv([("I. TOTAL OTHER COSTS (E + F + G)", money(I_TOTAL)),
     ("J. TOTAL CLOSING COSTS (D + I)", money(J_TOTAL))])}
<h2>Calculating Cash to Close</h2>
{kv([("Total Closing Costs (J)", money(J_TOTAL)), ("Down Payment / Funds from Borrower", money(DOWN)),
     ("Deposit (earnest money)", "-" + money(LOAN['deposit'])),
     ("Estimated Cash to Close", money(CASH_TO_CLOSE))])}
<h2>Comparisons</h2>
{kv([("Annual Percentage Rate (APR)", APR), ("Loan-to-Value", f"{LTV:.0f}%")])}
"""

title = f"""
<h1>{TITLE['company']}</h1>
<p>1550 Howe Ave, Sacramento, CA 95825 &nbsp;|&nbsp; Title Officer: {TITLE['officer']}, {TITLE['officer_phone']}</p>
<h2>PRELIMINARY REPORT</h2>
{kv([("Order No.", TITLE['order']), ("Effective Date", f"{TITLE['effective']} at 7:30 a.m."),
     ("Property Address", PROP['addr']), ("APN", PROP['apn']),
     ("Proposed Insured (Lender)", LOAN['lender']), ("Proposed Loan Amount", money(LOAN['amount'])),
     ("Policies to be issued", "CLTA/ALTA Homeowner's Policy (2021); ALTA Loan Policy (2021)"),
     ("Title vested in", TITLE['vested'])])}
<h2>LEGAL DESCRIPTION</h2>
<p>Lot 17 of "Harbor View Estates Unit No. 3", according to the map thereof filed in the office of the County
Recorder of Sacramento County on March 2, 1994, in Book 231 of Maps, Page 14.</p>
<h2>EXCEPTIONS (Schedule B)</h2>
<p>1. General and special county taxes for fiscal year 2026-2027, a lien not yet due or payable.
First installment: $2,887.50. Second installment: $2,887.50.</p>
<p>2. An easement for public utilities and incidental purposes in favor of Sacramento Municipal Utility
District, recorded April 11, 1994 as Instrument No. 19940411-0823, affecting the rear 10 feet of said land.</p>
<p>3. A Deed of Trust to secure an indebtedness in the original amount of $312,000.00, recorded August 2, 2017
as Instrument No. 201708020456. Trustor: Thomas J. Albrecht and Linda M. Albrecht. Beneficiary: Wells Fargo
Bank, N.A. Trustee: First American Title Insurance Company.
<b>Requirement: this Deed of Trust is to be paid off and reconveyed at close of escrow.</b></p>
<p>4. Covenants, conditions and restrictions recorded March 2, 1994 as Instrument No. 19940302-0311.</p>
<h2>INFORMATIONAL NOTES</h2>
<p>A. No open deeds of trust or liens were found against the proposed buyer, {B['name']}, in a general index search.</p>
<p>B. Wire instructions will be provided only by phone from your title officer. Never act on emailed wire instructions.</p>
"""

files = [("01_loan_application.pdf", [app1, app2]),
         ("02_pay_slips.pdf", [payslip(*p) for p in PAYSLIPS]),
         ("03_bank_statement.pdf", [bank]),
         ("04_loan_estimate.pdf", [le1, le2]),
         ("05_preliminary_title_report.pdf", [title])]

def key():
    pages, start = {}, 1
    for f, p in files:
        pages[f] = (start, start + len(p) - 1)
        start += len(p)
    pg = lambda f, i=0: f"merged p.{pages[f][0] + i}"
    A_, P_, BK, L, T = [x for x, _ in files]
    q = [
        ("What gross monthly income is stated on the loan application?", money(STATED_MONTHLY), f"Loan Application, {pg(A_)}"),
        ("Who is the borrower's current employer?", EMP['name'], f"Loan Application, {pg(A_)} / Pay Slip"),
        ("What is the borrower's start date with the current employer?", EMP['start'], f"Loan Application, {pg(A_)}"),
        ("What is the net pay on the pay slip for the period ending 07/31/2026?", money(NET), f"Pay Slip, {pg(P_, 1)}"),
        ("What is the gross pay per pay period on the pay slips?", money(SEMI_GROSS), f"Pay Slip, {pg(P_)}"),
        ("What is the annual base salary on the pay slip?", money(SEMI_GROSS * 24), f"Pay Slip, {pg(P_)}"),
        ("What is the year-to-date gross on the 07/31/2026 pay slip?", money(SEMI_GROSS * 14), f"Pay Slip, {pg(P_, 1)}"),
        ("How much 401(k) is deducted per pay period?", money(deductions[4][1]), f"Pay Slip, {pg(P_)}"),
        ("What is the ending balance on the bank statement?", money(END_BAL), f"Bank Statement, {pg(BK)}"),
        ("What was the largest single deposit on the bank statement, and what was it?", f"{money(12000)}, transfer in from account ***9902 on 07/12", f"Bank Statement, {pg(BK)}"),
        ("What checking balance did the applicant list on the application?", money(52400), f"Loan Application, {pg(A_)}"),
        ("What is the monthly payment on the auto loan?", money(AUTO['monthly']), f"Loan Application {pg(A_)} / Bank Statement"),
        ("What is the loan amount?", money(LOAN['amount']), f"Loan Estimate {pg(L)} / Loan Application {pg(A_, 1)}"),
        ("What is the interest rate?", f"{LOAN['rate']:.3f}%", f"Loan Estimate, {pg(L)}"),
        ("What is the monthly principal and interest?", money(PI), f"Loan Estimate, {pg(L)}"),
        ("Until when is the rate locked?", f"{LOAN['lock_until']} at 5:00 p.m. PT", f"Loan Estimate, {pg(L)}"),
        ("What is the estimated total monthly payment?", money(PI + ESCROW_MO), f"Loan Estimate, {pg(L)}"),
        ("What are the total closing costs?", money(J_TOTAL), f"Loan Estimate, {pg(L)} / {pg(L, 1)}"),
        ("What is the estimated cash to close?", money(CASH_TO_CLOSE), f"Loan Estimate, {pg(L)} / {pg(L, 1)}"),
        ("What is the origination fee?", money(A[0][1]), f"Loan Estimate, {pg(L, 1)}"),
        ("What is the prepaid interest amount?", money(PREPAID_INT), f"Loan Estimate, {pg(L, 1)}"),
        ("What is the purchase price of the property?", money(PROP['price']), f"Loan Estimate {pg(L)} / Loan Application {pg(A_, 1)}"),
        ("Who is title currently vested in?", TITLE['vested'], f"Preliminary Title Report, {pg(T)}"),
        ("Is there an easement on the property? To whom?", "Yes — public utilities, Sacramento Municipal Utility District, rear 10 feet", f"Preliminary Title Report, {pg(T)}"),
        ("Is there an existing deed of trust on the property, and for how much?", "Yes — $312,000.00 to Wells Fargo Bank, N.A.; to be paid off at close of escrow", f"Preliminary Title Report, {pg(T)}"),
        ("What is the title order number?", TITLE['order'], f"Preliminary Title Report, {pg(T)}"),
        ("What is the APN?", PROP['apn'], f"Preliminary Title Report, {pg(T)}"),
    ]
    not_found = [
        "What is the co-borrower's name?  (application says individual credit — no co-borrower)",
        "What is the borrower's credit score?",
        "What was the borrower's previous employer?",
        "What is the monthly HOA fee?",
        "What is the square footage of the property?",
        "What is the borrower's driver's license number?",
        "What is the amount on the gift letter?  (no gift letter; application says gifts: None)",
        "What is the appraised value of the property?  (appraisal FEE is listed; no appraisal report)",
    ]
    checks = [
        ("Income", f"Application states {money(STATED_MONTHLY)}/month; pay slips show {money(SEMI_GROSS)} semi-monthly = "
                   f"{money(SEMI_GROSS * 2)}/month (annual {money(SEMI_GROSS * 24)}). Overstated by {money(STATED_MONTHLY - SEMI_GROSS * 2)}/month."),
        ("Large deposit", f"Bank statement shows a {money(12000)} transfer in on 07/12 from account ***9902. Declaration C "
                          "(money from another party not disclosed) is answered NO and gifts are 'None' — needs a letter of explanation."),
        ("Net pay vs deposits", f"Both payroll deposits on the bank statement ({money(NET)}) match the pay slips' net pay — consistent."),
        ("Checking balance", f"Application lists {money(52400)}; statement ending balance is {money(END_BAL)} — consistent."),
        ("Seller's lien", "Title report exception 3: $312,000 deed of trust (Wells Fargo) must be paid off and reconveyed at closing."),
    ]
    lines = ["# Answer key — synthetic borrower file", "",
             "SYNTHETIC TEST DATA. Page numbers are of `whitfield_loan_packet.pdf` — the same "
             "as uploading the five separate PDFs together in filename order (01 → 05):", ""]
    lines += [f"- `{f}` → pages {a}–{b}" if a != b else f"- `{f}` → page {a}" for f, (a, b) in pages.items()]
    lines += ["", "## Answerable (expected answer — source)", ""]
    lines += [f"{i}. **{qq}**  \n   {ans} — *{src}*" for i, (qq, ans, src) in enumerate(q, 1)]
    lines += ["", "## Not in the file — the correct answer is \"not found\"", ""]
    lines += [f"{i}. {qq}" for i, qq in enumerate(not_found, len(q) + 1)]
    lines += ["", "## Planted issues — what a reviewer should find", ""]
    lines += [f"- **{k}:** {v}" for k, v in checks]
    lines += ["", "Classifier expectation: the two application pages → **Loan Application**; pay slips → Pay Slip; "
              "statement → Bank Statement; Loan Estimate → Lender Fee Sheet; title → Preliminary Title Report.", ""]
    with open(os.path.join(OUT, "answer_key.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return len(q), len(not_found)


if __name__ == "__main__":
    total = 0
    for f, pages in files:
        n_ = pdf(f, pages)
        total += n_
        print(f"{f:36} {n_} page(s)  {os.path.getsize(os.path.join(OUT, f)) / 1024:6.1f} KB")
    packet = fitz.open()
    for f, _ in files:
        with fitz.open(os.path.join(OUT, f)) as part:
            packet.insert_pdf(part)
    packet.save(os.path.join(OUT, "whitfield_loan_packet.pdf"), garbage=3, deflate=True)
    print(f"{'whitfield_loan_packet.pdf':36} {packet.page_count} page(s)  "
          f"{os.path.getsize(os.path.join(OUT, 'whitfield_loan_packet.pdf')) / 1024:6.1f} KB  <- the packet")
    a, nf = key()
    print(f"total {total} pages; answer_key.md: {a} answerable + {nf} not-found questions")
