# Answer key — synthetic borrower file

SYNTHETIC TEST DATA. Page numbers are of `whitfield_loan_packet.pdf` — the same as uploading the five separate PDFs together in filename order (01 → 05):

- `01_loan_application.pdf` → pages 1–2
- `02_pay_slips.pdf` → pages 3–4
- `03_bank_statement.pdf` → page 5
- `04_loan_estimate.pdf` → pages 6–7
- `05_preliminary_title_report.pdf` → page 8

## Answerable (expected answer — source)

1. **What gross monthly income is stated on the loan application?**  
   $9,500.00 — *Loan Application, merged p.1*
2. **Who is the borrower's current employer?**  
   Sierra Ridge Logistics, Inc. — *Loan Application, merged p.1 / Pay Slip*
3. **What is the borrower's start date with the current employer?**  
   06/01/2019 — *Loan Application, merged p.1*
4. **What is the net pay on the pay slip for the period ending 07/31/2026?**  
   $2,845.31 — *Pay Slip, merged p.4*
5. **What is the gross pay per pay period on the pay slips?**  
   $4,375.00 — *Pay Slip, merged p.3*
6. **What is the annual base salary on the pay slip?**  
   $105,000.00 — *Pay Slip, merged p.3*
7. **What is the year-to-date gross on the 07/31/2026 pay slip?**  
   $61,250.00 — *Pay Slip, merged p.4*
8. **How much 401(k) is deducted per pay period?**  
   $218.75 — *Pay Slip, merged p.3*
9. **What is the ending balance on the bank statement?**  
   $52,489.98 — *Bank Statement, merged p.5*
10. **What was the largest single deposit on the bank statement, and what was it?**  
   $12,000.00, transfer in from account ***9902 on 07/12 — *Bank Statement, merged p.5*
11. **What checking balance did the applicant list on the application?**  
   $52,400.00 — *Loan Application, merged p.1*
12. **What is the monthly payment on the auto loan?**  
   $486.00 — *Loan Application merged p.1 / Bank Statement*
13. **What is the loan amount?**  
   $420,000.00 — *Loan Estimate merged p.6 / Loan Application merged p.2*
14. **What is the interest rate?**  
   6.375% — *Loan Estimate, merged p.6*
15. **What is the monthly principal and interest?**  
   $2,620.25 — *Loan Estimate, merged p.6*
16. **Until when is the rate locked?**  
   09/15/2026 at 5:00 p.m. PT — *Loan Estimate, merged p.6*
17. **What is the estimated total monthly payment?**  
   $3,226.50 — *Loan Estimate, merged p.6*
18. **What are the total closing costs?**  
   $11,133.59 — *Loan Estimate, merged p.6 / merged p.7*
19. **What is the estimated cash to close?**  
   $106,133.59 — *Loan Estimate, merged p.6 / merged p.7*
20. **What is the origination fee?**  
   $1,050.00 — *Loan Estimate, merged p.7*
21. **What is the prepaid interest amount?**  
   $1,100.34 — *Loan Estimate, merged p.7*
22. **What is the purchase price of the property?**  
   $525,000.00 — *Loan Estimate merged p.6 / Loan Application merged p.2*
23. **Who is title currently vested in?**  
   Thomas J. Albrecht and Linda M. Albrecht, husband and wife, as joint tenants — *Preliminary Title Report, merged p.8*
24. **Is there an easement on the property? To whom?**  
   Yes — public utilities, Sacramento Municipal Utility District, rear 10 feet — *Preliminary Title Report, merged p.8*
25. **Is there an existing deed of trust on the property, and for how much?**  
   Yes — $312,000.00 to Wells Fargo Bank, N.A.; to be paid off at close of escrow — *Preliminary Title Report, merged p.8*
26. **What is the title order number?**  
   2026-04471-SR — *Preliminary Title Report, merged p.8*
27. **What is the APN?**  
   031-0482-017-0000 — *Preliminary Title Report, merged p.8*

## Not in the file — the correct answer is "not found"

28. What is the co-borrower's name?  (application says individual credit — no co-borrower)
29. What is the borrower's credit score?
30. What was the borrower's previous employer?
31. What is the monthly HOA fee?
32. What is the square footage of the property?
33. What is the borrower's driver's license number?
34. What is the amount on the gift letter?  (no gift letter; application says gifts: None)
35. What is the appraised value of the property?  (appraisal FEE is listed; no appraisal report)

## Planted issues — what a reviewer should find

- **Income:** Application states $9,500.00/month; pay slips show $4,375.00 semi-monthly = $8,750.00/month (annual $105,000.00). Overstated by $750.00/month.
- **Large deposit:** Bank statement shows a $12,000.00 transfer in on 07/12 from account ***9902. Declaration C (money from another party not disclosed) is answered NO and gifts are 'None' — needs a letter of explanation.
- **Net pay vs deposits:** Both payroll deposits on the bank statement ($2,845.31) match the pay slips' net pay — consistent.
- **Checking balance:** Application lists $52,400.00; statement ending balance is $52,489.98 — consistent.
- **Seller's lien:** Title report exception 3: $312,000 deed of trust (Wells Fargo) must be paid off and reconveyed at closing.

Classifier expectation: the two application pages → **Loan Application**; pay slips → Pay Slip; statement → Bank Statement; Loan Estimate → Lender Fee Sheet; title → Preliminary Title Report.
