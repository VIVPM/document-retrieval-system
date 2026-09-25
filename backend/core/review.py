"""review.py — the automatic file review: extract fields, compare them, flag issues."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

INCOME_TOLERANCE = 0.05
BALANCE_TOLERANCE = 0.10
LARGE_DEPOSIT_SHARE = 0.25
AMOUNT_EPS = 1.00

SCHEMAS: dict[str, dict] = {
    "Loan Application": {
        "fields": {
            "borrower_name": ("text", "Borrower's full name", ["name", "borrower name", "applicant name"]),
            "employer": ("text", "Current employer or business name", ["employer or business name", "employer name", "current employer", "employer"]),
            "employment_start": ("date", "Start date with the current employer", ["start date", "date of joining"]),
            "stated_monthly_income": ("money", "TOTAL gross monthly income the applicant states", ["total gross monthly income", "total monthly income", "gross monthly income", "monthly income"]),
            "checking_balance": ("money", "Stated balance of the checking/savings account", []),
            "loan_amount": ("money", "Loan amount requested", ["loan amount", "loan amount requested"]),
            "purchase_price": ("money", "Property value / purchase price", ["property value purchase price", "purchase price", "property value"]),
            "undisclosed_funds": ("bool", "Answer to the declaration asking whether the applicant is borrowing or obtaining money from another party not disclosed on the application (YES/NO)", []),
            "gifts": ("text", "Gifts or grants the applicant has been given or will receive", ["gifts or grants you have been given or will receive"]),
            "co_borrower": ("text", "Co-borrower / co-applicant name, or 'none' if individual credit", []),
        },
        "lists": {
            "liabilities": "Each liability: creditor, monthly_payment",
        },
    },
    "Pay Slip": {
        "fields": {
            "employer": ("text", "Employer / company name", ["company", "employer", "company name"]),
            "employee_name": ("text", "Employee name", ["employee name", "employee"]),
            "pay_frequency": ("text", "Pay frequency (weekly, bi-weekly, semi-monthly, monthly)", ["pay frequency", "pay cycle"]),
            "period_start": ("date", "Start date of the LATEST pay period", []),
            "period_end": ("date", "End date of the LATEST pay period", []),
            "gross_pay": ("money", "Gross pay for the LATEST pay period (current, not year-to-date)", ["gross earnings", "gross pay", "current gross"]),
            "net_pay": ("money", "Net pay for the LATEST pay period (current, not year-to-date)", ["net pay", "take home pay"]),
            "ytd_gross": ("money", "Year-to-date gross on the LATEST pay slip", ["ytd gross", "year to date gross"]),
        },
        "lists": {},
    },
    "Bank Statement": {
        "fields": {
            "account_holder": ("text", "Account holder name", ["member", "account holder", "name"]),
            "ending_balance": ("money", "Ending balance for the statement period", ["ending balance", "closing balance"]),
            "beginning_balance": ("money", "Beginning balance for the statement period", ["beginning balance", "opening balance"]),
        },
        "lists": {
            "deposits": "Each deposit/credit: date, description, amount",
            "debits": "Each withdrawal/debit: date, description, amount",
        },
    },
    "Lender Fee Sheet": {
        "fields": {
            "loan_amount": ("money", "Loan amount", ["loan amount"]),
            "interest_rate": ("text", "Interest rate", ["interest rate"]),
            "sale_price": ("money", "Sale price / purchase price of the property", ["sale price", "purchase price"]),
            "monthly_pi": ("money", "Monthly principal & interest", ["monthly principal interest", "principal interest"]),
            "cash_to_close": ("money", "Estimated cash to close", ["estimated cash to close", "cash to close"]),
            "closing_costs": ("money", "Total / estimated closing costs", ["estimated closing costs", "total closing costs"]),
        },
        "lists": {},
    },
    "Preliminary Title Report": {
        "fields": {
            "vested_owner": ("text", "Who title is currently vested in", ["title vested in", "vested in"]),
        },
        "lists": {
            "liens": "Each deed of trust / mortgage / lien: amount, holder, payoff_required (true if the report says it is to be paid off or reconveyed)",
            "easements": "Each easement: holder, purpose",
        },
    },
}
REQUIRED_TYPES = ("Loan Application", "Pay Slip", "Bank Statement")


def norm_label(s: str) -> str:
    """'TOTAL Gross Monthly Income:' -> 'total gross monthly income'."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split())


def parse_money(s) -> float | None:
    """'$9,500.00' -> 9500.0; '-$2,150.00' / '(2,150.00)' -> -2150.0; else None."""
    if isinstance(s, (int, float)):
        return float(s)
    if not s:
        return None
    t = str(s).strip()
    neg = t.startswith("-") or t.startswith("(") or "-$" in t
    m = re.search(r"\d[\d,]*(?:\.\d+)?", t)
    if not m:
        return None
    v = float(m.group(0).replace(",", ""))
    return -v if neg else v


def parse_bool(s) -> bool | None:
    t = str(s or "").strip().lower()
    if t in ("yes", "y", "true", "[x] yes"):
        return True
    if t in ("no", "n", "false", "[x] no"):
        return False
    return None


def _parse(kind: str, raw):
    if kind == "money":
        return parse_money(raw)
    if kind == "bool":
        return parse_bool(raw)
    return str(raw).strip() if raw not in (None, "") else None


_MONEY = re.compile(r"[-+(]?\$?\s?\d[\d,]*\.\d{2}\)?")


def _flat(s) -> str:
    """Lower-case, single-spaced, table cell separators dropped -- so a value
    quoted from a table row matches the row however its cells were joined."""
    return " ".join(str(s or "").replace("|", " ").split()).lower()


def _page_texts(doc) -> list[tuple[int, str]]:
    """[(page 1-based, normalised text)] for grounding and page lookup."""
    by_page: dict[int, list[str]] = {}
    for b in getattr(doc, "blocks", None) or []:
        by_page.setdefault(b.page_num, []).append(b.content)
    if not by_page:
        by_page = {doc.page_start: [doc.text]}
    return [(p + 1, _flat(" ".join(parts))) for p, parts in sorted(by_page.items())]


def _locate(raw, pages: list[tuple[int, str]]) -> int | None:
    """Page (1-based) whose text contains `raw` verbatim, or None if nowhere --
    that None IS the grounding guard. Located in code, never taken from the model."""
    r = _flat(raw)
    if not r:
        return None
    return next((p for p, t in pages if r in t), None)


def _payoff_noted(amount_raw, pages) -> bool:
    """Decided from the report's own words near the lien amount, not by the
    model: does the same passage say it is paid off or reconveyed?"""
    r = _flat(amount_raw)
    for _p, t in pages:
        i = t.find(r) if r else -1
        if i >= 0 and re.search(r"paid off|payoff|pay off|reconvey", t[i:i + 500]):
            return True
    return False


def _doc_text(doc) -> str:
    """Document text with page markers (1-based, merged-file page numbers)."""
    by_page: dict[int, list[str]] = {}
    for b in getattr(doc, "blocks", None) or []:
        by_page.setdefault(b.page_num, []).append(b.content)
    if not by_page:
        return f"=== PAGE {doc.page_start + 1} ===\n{doc.text}"
    return "\n\n".join(f"=== PAGE {p + 1} ===\n" + "\n".join(parts)
                       for p, parts in sorted(by_page.items()))


def fields_from_forms(doc_type: str, kv) -> dict:
    """Map Textract label -> value pairs onto the schema. Later pages win, so a
    multi-page pay-slip document reports its latest slip."""
    spec = SCHEMAS[doc_type]["fields"]
    out: dict = {}
    for key, value, page in kv or []:
        k = norm_label(key)
        for name, (kind, _desc, syns) in spec.items():
            if k in syns:
                raw = value
                if kind == "money":
                    m = _MONEY.search(value)
                    raw = m.group(0).strip() if m else value
                v = _parse(kind, raw)
                if v is not None:
                    out[name] = {"value": v, "raw": raw, "page": page + 1, "source": "form"}
    return out


def _llm_prompt(doc_type: str, want_fields: dict, want_lists: dict, text: str) -> str:
    f_lines = "\n".join(f'  "{n}": {d}' for n, (_k, d, _s) in want_fields.items())
    l_lines = "\n".join(f'  "{n}": {d}' for n, d in want_lists.items())
    return f"""You extract fields from ONE {doc_type} for a loan file review.

Rules:
- Copy every value EXACTLY as printed in the document (same digits, symbols,
  punctuation). Do not compute, round, reformat or infer.
- If a field is not present, use null. Never guess.
- "page" is the number from the nearest "=== PAGE n ===" marker above the value.

Return ONLY JSON, no prose, in this shape:
{{"fields": {{"<name>": {{"value": "<exact text>", "page": <n>}} or null}},
  "lists": {{"<name>": [{{"<attr>": "<exact text>", ..., "page": <n>}}]}}}}

Fields:
{f_lines or "  (none)"}

Lists (each item's amounts copied exactly):
{l_lines or "  (none)"}

Document:
{text}
"""


def _loads(s: str) -> dict:
    """Parse the model's JSON, tolerating code fences, surrounding prose and
    trailing commas. Raises ValueError if it still is not JSON."""
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    m = re.search(r"\{.*\}", s, re.S)
    s = m.group(0) if m else s
    for candidate in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("model reply is not valid JSON")


def _default_complete(prompt: str) -> str:
    from llm.llm_router import llm
    return llm.complete(prompt, temperature=0, fast=True, thinking_budget=0,
                        max_tokens=4096).text or ""


def fields_from_llm(doc_type: str, doc, missing: dict, complete=None) -> tuple[dict, dict, int]:
    """One LLM call for the missing scalar fields and all list fields.
    Returns (fields, lists, dropped) -- `dropped` counts values rejected by the
    grounding guard."""
    lists_spec = SCHEMAS[doc_type]["lists"]
    if not missing and not lists_spec:
        return {}, {}, 0
    if complete is None:
        complete = _default_complete
    pages = _page_texts(doc)
    prompt = _llm_prompt(doc_type, missing, lists_spec, _doc_text(doc))
    try:
        data = _loads(complete(prompt))
    except ValueError:
        data = _loads(complete(prompt + "\n\nYour previous reply was not valid JSON. "
                                        "Reply again with ONLY valid JSON in the shape above."))

    fields, dropped = {}, 0
    for name, item in (data.get("fields") or {}).items():
        if name not in missing:
            continue
        raw = item.get("value") if isinstance(item, dict) else item
        if raw in (None, "") or isinstance(raw, (list, dict)):
            continue
        page = _locate(raw, pages)
        if page is None:
            dropped += 1
            continue
        v = _parse(missing[name][0], raw)
        if v is not None:
            fields[name] = {"value": v, "raw": str(raw), "page": page, "source": "llm"}

    lists = {}
    for name in lists_spec:
        kept = []
        for it in (data.get("lists") or {}).get(name) or []:
            if not isinstance(it, dict):
                continue
            it = {k: v.get("value") if isinstance(v, dict) and "value" in v else v
                  for k, v in it.items()}
            amount_raw = it.get("amount") or it.get("monthly_payment")
            if name == "liens" and amount_raw in (None, ""):
                continue
            if amount_raw:
                page = _locate(amount_raw, pages)
                if page is None:
                    dropped += 1
                    continue
            else:
                text_vals = [v for v in it.values() if isinstance(v, str) and len(v) > 3]
                page = next((pg for pg in (_locate(v, pages) for v in text_vals) if pg), None)
            row = dict(it, page=page)
            for k in ("amount", "monthly_payment"):
                if k in row:
                    row[k + "_value"] = parse_money(row[k])
            if name == "liens":
                row["payoff_required"] = _payoff_noted(amount_raw, pages)
            kept.append(row)
        lists[name] = kept
    return fields, lists, dropped


def extract_document(doc, complete=None) -> dict | None:
    """All fields for one logical document, or None if its type has no schema."""
    if doc.doc_type not in SCHEMAS:
        return None
    spec = SCHEMAS[doc.doc_type]["fields"]
    fields = fields_from_forms(doc.doc_type, getattr(doc, "kv", None))
    missing = {n: s for n, s in spec.items() if n not in fields}
    llm_fields, lists, dropped = fields_from_llm(doc.doc_type, doc, missing, complete)
    fields.update(llm_fields)
    return {"doc_type": doc.doc_type, "pages": [doc.page_start + 1, doc.page_end + 1],
            "fields": fields, "lists": lists, "ungrounded_dropped": dropped}


def _ev(doc: dict | None, field: str, label: str) -> dict:
    f = (doc or {}).get("fields", {}).get(field)
    return {"doc_type": (doc or {}).get("doc_type"), "label": label,
            "value": f["raw"] if f else None, "page": f["page"] if f else None}


def _val(doc, field):
    f = (doc or {}).get("fields", {}).get(field)
    return f["value"] if f else None


def _norm_company(s) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower())
    drop = {"inc", "llc", "ltd", "limited", "corp", "corporation", "co", "company", "pvt", "private", "the"}
    return " ".join(w for w in s.split() if w not in drop)


def periods_per_month(freq: str | None, start: str | None, end: str | None) -> float | None:
    f = (freq or "").lower().replace("-", "").replace(" ", "")
    table = {"weekly": 52 / 12, "biweekly": 26 / 12, "fortnightly": 26 / 12,
             "semimonthly": 2.0, "twicemonthly": 2.0, "monthly": 1.0}
    for k, v in table.items():
        if f == k:
            return v
    try:
        a, b = (datetime.strptime(str(x), "%m/%d/%Y") for x in (start, end))
        days = (b - a).days + 1
    except Exception:
        return None
    if days <= 8:
        return 52 / 12
    if days <= 14:
        return 26 / 12
    if days <= 16:
        return 2.0
    if days <= 31:
        return 1.0
    return None


def _check(cid, label, status, detail, evidence):
    return {"id": cid, "label": label, "status": status, "detail": detail, "evidence": evidence}


def _money(v):
    return f"${v:,.2f}"


def _n(k: int, word: str) -> str:
    """'1 deposit', '2 deposits'."""
    return f"{k} {word}" + ("" if k == 1 else "s")


def _end(s) -> str:
    """Strip a trailing period, so a value like 'Inc.' can end a sentence."""
    return str(s).rstrip(".")


def run_rules(docs: list[dict]) -> list[dict]:
    """Compare extracted fields across documents. Pure function of `docs`."""
    first = {}
    for d in docs:
        first.setdefault(d["doc_type"], d)
    slips = [d for d in docs if d["doc_type"] == "Pay Slip"]
    app, bank = first.get("Loan Application"), first.get("Bank Statement")
    slip = slips[-1] if slips else None
    le, title = first.get("Lender Fee Sheet"), first.get("Preliminary Title Report")
    checks = []

    for t in REQUIRED_TYPES:
        if t not in first:
            checks.append(_check(f"missing_{t.lower().replace(' ', '_')}", f"{t} present",
                                 "missing", f"No {t} found in the packet.", []))

    stated = _val(app, "stated_monthly_income")
    gross = _val(slip, "gross_pay")
    ppm = periods_per_month(_val(slip, "pay_frequency"), _val(slip, "period_start"),
                            _val(slip, "period_end")) if slip else None
    verified = round(gross * ppm, 2) if gross and ppm else None
    ev = [_ev(app, "stated_monthly_income", "Stated monthly income"),
          _ev(slip, "gross_pay", "Gross pay per period")]
    if stated is None or verified is None:
        checks.append(_check("income", "Income", "missing",
                             "Could not read " + ("stated income" if stated is None else
                                                  "pay-slip gross or pay frequency") + ".", ev))
    else:
        diff = stated - verified
        ok = abs(diff) <= INCOME_TOLERANCE * verified
        checks.append(_check("income", "Income", "match" if ok else "mismatch",
                             f"Stated {_money(stated)}/month; pay slips support {_money(verified)}/month"
                             + ("" if ok else f" — {'over' if diff > 0 else 'under'}stated by {_money(abs(diff))}/month")
                             + ".", ev))

    a_emp, s_emp = _val(app, "employer"), _val(slip, "employer")
    ev = [_ev(app, "employer", "Employer (application)"), _ev(slip, "employer", "Employer (pay slip)")]
    if not a_emp or not s_emp:
        checks.append(_check("employer", "Employer", "missing", "Employer not found on both documents.", ev))
    else:
        na, ns = _norm_company(a_emp), _norm_company(s_emp)
        ok = na == ns or na in ns or ns in na
        checks.append(_check("employer", "Employer", "match" if ok else "mismatch",
                             f"Application: {a_emp}; pay slip: {_end(s_emp)}.", ev))

    a_bal, e_bal = _val(app, "checking_balance"), _val(bank, "ending_balance")
    ev = [_ev(app, "checking_balance", "Stated balance"), _ev(bank, "ending_balance", "Statement ending balance")]
    if a_bal is None or e_bal is None:
        checks.append(_check("balance", "Checking balance", "missing", "Balance not found on both documents.", ev))
    else:
        ok = abs(a_bal - e_bal) <= BALANCE_TOLERANCE * max(e_bal, 1)
        checks.append(_check("balance", "Checking balance", "match" if ok else "mismatch",
                             f"Application {_money(a_bal)}; statement ends at {_money(e_bal)}.", ev))

    net = _val(slip, "net_pay")
    deposits = (bank or {}).get("lists", {}).get("deposits", [])
    employer_words = set(_norm_company(s_emp or a_emp).split())

    def is_payroll(dep):
        desc = str(dep.get("description", "")).lower()
        amt = dep.get("amount_value")
        return (net is not None and amt is not None and abs(abs(amt) - net) <= AMOUNT_EPS) \
            or any(w in desc for w in ("payroll", "salary", "direct dep")) \
            or (employer_words and employer_words <= set(_norm_company(desc).split()))

    if bank is not None:
        payroll = [d for d in deposits if is_payroll(d)]
        if net is not None:
            hits = [d for d in payroll if d.get("amount_value") is not None
                    and abs(abs(d["amount_value"]) - net) <= AMOUNT_EPS]
            checks.append(_check("payroll", "Payroll deposits", "match" if hits else "review",
                                 (f"{_n(len(hits), 'deposit')} match{'es' if len(hits) == 1 else ''} pay-slip net pay {_money(net)}." if hits
                                  else f"No deposit matches pay-slip net pay {_money(net)}."),
                                 [_ev(slip, "net_pay", "Net pay")] +
                                 [{"doc_type": "Bank Statement", "label": d.get("description"),
                                   "value": d.get("amount"), "page": d.get("page")} for d in hits]))
        income = verified or stated
        if income:
            big = [d for d in deposits if d not in payroll and d.get("amount_value") is not None
                   and abs(d["amount_value"]) > LARGE_DEPOSIT_SHARE * income]
            undisclosed = _val(app, "undisclosed_funds")
            ev = [{"doc_type": "Bank Statement", "label": f"{d.get('date', '')} {d.get('description', '')}".strip(),
                   "value": d.get("amount"), "page": d.get("page")} for d in big]
            if big:
                note = ""
                if undisclosed is False:
                    note = " Declaration says no undisclosed funds from another party."
                    ev.append(_ev(app, "undisclosed_funds", "Declaration: undisclosed funds"))
                checks.append(_check("large_deposits", "Large deposits", "review",
                                     f"{_n(len(big), 'non-payroll deposit')} above {int(LARGE_DEPOSIT_SHARE * 100)}% of "
                                     f"monthly income ({_money(income)}) — source needs explaining.{note}", ev))
            else:
                checks.append(_check("large_deposits", "Large deposits", "match",
                                     "No unexplained large deposits.", []))

    for cid, label, af, lf in (("loan_amount", "Loan amount", "loan_amount", "loan_amount"),
                               ("price", "Purchase price", "purchase_price", "sale_price")):
        av, lv = _val(app, af), _val(le, lf)
        ev = [_ev(app, af, f"{label} (application)"), _ev(le, lf, f"{label} (loan estimate)")]
        if av is None or lv is None:
            if app is not None and le is not None:
                checks.append(_check(cid, label, "missing", f"{label} not found on both documents.", ev))
            continue
        ok = abs(av - lv) <= AMOUNT_EPS
        checks.append(_check(cid, label, "match" if ok else "mismatch",
                             f"Application {_money(av)}; loan estimate {_money(lv)}.", ev))

    liabilities = (app or {}).get("lists", {}).get("liabilities", [])
    debits = (bank or {}).get("lists", {}).get("debits", [])
    if liabilities and bank is not None:
        missing_pay, ev = [], []
        for li in liabilities:
            pay = li.get("monthly_payment_value")
            if pay is None:
                continue
            hit = next((d for d in debits if d.get("amount_value") is not None
                        and abs(abs(d["amount_value"]) - pay) <= AMOUNT_EPS), None)
            if hit is None:
                words = set(_norm_company(li.get("creditor")).split()) - {"card", "services", "bank"}
                hit = next((d for d in debits if words and words
                            <= set(_norm_company(d.get("description")).split())), None)
            ev.append({"doc_type": "Loan Application", "label": li.get("creditor"),
                       "value": li.get("monthly_payment"), "page": li.get("page")})
            if hit:
                ev.append({"doc_type": "Bank Statement", "label": hit.get("description"),
                           "value": hit.get("amount"), "page": hit.get("page")})
            else:
                missing_pay.append(li.get("creditor") or _money(pay))
        checks.append(_check("liabilities", "Stated debt payments",
                             "review" if missing_pay else "match",
                             ("Not seen on the statement: " + ", ".join(missing_pay) + "." if missing_pay
                              else "Every stated monthly payment appears as a statement debit."), ev))

    if title is not None:
        liens = title.get("lists", {}).get("liens", [])
        for li in liens:
            payoff = str(li.get("payoff_required", "")).lower() in ("true", "yes", "1")
            checks.append(_check("lien", "Lien on title", "review" if payoff else "mismatch",
                                 f"{li.get('amount', '?')} to {li.get('holder', '?')} — "
                                 + ("must be paid off at closing." if payoff else "no payoff noted; must be resolved."),
                                 [{"doc_type": "Preliminary Title Report", "label": li.get("holder"),
                                   "value": li.get("amount"), "page": li.get("page")}]))
        for e in title.get("lists", {}).get("easements", []):
            checks.append(_check("easement", "Easement", "info",
                                 f"{str(e.get('purpose') or 'Easement').capitalize()} — {_end(e.get('holder', '?'))}.",
                                 [{"doc_type": "Preliminary Title Report", "label": e.get("holder"),
                                   "value": e.get("purpose"), "page": e.get("page")}]))
    return checks


def split_by_page_type(logical_docs) -> list:
    """Boundary detection can merge two documents -- a pay slip ending
    "Deposited to: Golden State Credit Union" ran straight into that bank's
    statement. For the REVIEW only (search keeps its own grouping), start a new
    document wherever a page's own classification is a different reviewable
    type. Continuation pages classified as something unreviewable stay put."""
    from types import SimpleNamespace as NS
    out = []
    for d in logical_docs:
        types = getattr(d, "page_types", None) or []
        if len(types) != d.page_end - d.page_start + 1:
            out.append(d)
            continue
        cur_type, cur_start, parts = d.doc_type, d.page_start, []
        for i, t in enumerate(types):
            if i and t in SCHEMAS and t != cur_type:
                parts.append((cur_type, cur_start, d.page_start + i - 1))
                cur_type, cur_start = t, d.page_start + i
        if not parts:
            out.append(d)
            continue
        parts.append((cur_type, cur_start, d.page_end))
        for t, a, b in parts:
            out.append(NS(doc_type=t, page_start=a, page_end=b, text="",
                          blocks=[x for x in d.blocks if a <= x.page_num <= b],
                          kv=[k for k in (d.kv or []) if a <= k[2] <= b]))
    return out


def build_review(logical_docs, complete=None) -> dict:
    """Extract (documents in parallel) and compare. Never raises for one bad
    document -- it is reported and the rest of the review still runs."""
    logical_docs = split_by_page_type(logical_docs)
    docs, errors = [], []

    def one(d):
        try:
            return extract_document(d, complete)
        except Exception as e:
            errors.append(f"{d.doc_type} p.{d.page_start + 1}: {type(e).__name__}: {e}"[:300])
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        docs = [r for r in pool.map(one, logical_docs) if r]
    checks = run_rules(docs)
    order = {"mismatch": 0, "review": 1, "missing": 2, "match": 3, "info": 4}
    checks.sort(key=lambda c: order.get(c["status"], 9))
    summary = {k: sum(c["status"] == k for c in checks) for k in order}
    app = next((d for d in docs if d["doc_type"] == "Loan Application"), None)
    return {
        "status": "done",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "borrower": _val(app, "borrower_name"),
        "summary": summary,
        "checks": checks,
        "documents": docs,
        "errors": errors,
    }


if __name__ == "__main__":
    from types import SimpleNamespace as NS

    assert parse_money("$9,500.00") == 9500.0
    assert parse_money("-$2,150.00") == -2150.0 and parse_money("(1,234.50)") == -1234.5
    assert parse_money("+$12,000.00") == 12000.0 and parse_money("n/a") is None
    assert norm_label("TOTAL Gross Monthly Income:") == "total gross monthly income"
    assert abs(periods_per_month(None, "07/16/2026", "07/31/2026") - 2.0) < 1e-9
    assert abs(periods_per_month("Bi-Weekly", None, None) - 26 / 12) < 1e-9

    f = fields_from_forms("Pay Slip", [("Net Pay:", "$2,000.00", 2), ("NET PAY", "$2,845.31", 3)])
    assert f["net_pay"]["value"] == 2845.31 and f["net_pay"]["page"] == 4 and f["net_pay"]["source"] == "form"

    doc = NS(doc_type="Pay Slip", page_start=2, page_end=2, text="", kv=[],
             blocks=[NS(page_num=2, content="Gross Earnings | $4,375.00\nNET PAY $2,845.31")])
    def fake(_prompt):
        return json.dumps({"fields": {"gross_pay": "$4,375.00", "net_pay": "$9,999.99"}})
    got, _lists, dropped = fields_from_llm("Pay Slip", doc, {k: SCHEMAS["Pay Slip"]["fields"][k]
                                                              for k in ("gross_pay", "net_pay")}, fake)
    assert got["gross_pay"]["value"] == 4375.0 and got["gross_pay"]["page"] == 3
    assert "net_pay" not in got and dropped == 1

    assert _loads('```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    replies = iter(["{broken", '{"fields": {"gross_pay": "$4,375.00"}}'])

    def flaky(_prompt):
        return next(replies)
    got, _l, _d = fields_from_llm("Pay Slip", doc, {"gross_pay": SCHEMAS["Pay Slip"]["fields"]["gross_pay"]},
                                  flaky)
    assert got["gross_pay"]["value"] == 4375.0

    f = fields_from_forms("Pay Slip", [("Gross Earnings", "$4,375.00 $61,250.00", 3)])
    assert f["gross_pay"]["raw"] == "$4,375.00", f["gross_pay"]

    tdoc = NS(doc_type="Preliminary Title Report", page_start=7, page_end=7, text="", kv=[],
              blocks=[NS(page_num=7, content="3. A Deed of Trust ... $312,000.00 ... Beneficiary: "
                                              "Wells Fargo. Requirement: this Deed of Trust is to be paid off.")])

    def fake_title(_prompt):
        return json.dumps({"fields": {}, "lists": {"liens": [
            {"amount": "$312,000.00", "holder": "Wells Fargo", "payoff_required": False},
            {"amount": None, "holder": "First American Title Insurance Company"}]}})
    _f, lists, _d = fields_from_llm("Preliminary Title Report", tdoc, {}, fake_title)
    assert [li["holder"] for li in lists["liens"]] == ["Wells Fargo"]
    assert lists["liens"][0]["payoff_required"] is True and lists["liens"][0]["page"] == 8

    merged = NS(doc_type="Pay Slip", page_start=2, page_end=4, text="", kv=[(("Net Pay", "$1.00", 3))],
                page_types=["Pay Slip", "Pay Slip", "Bank Statement"],
                blocks=[NS(page_num=p, content=str(p)) for p in (2, 3, 4)])
    parts = split_by_page_type([merged])
    assert [(p.doc_type, p.page_start, p.page_end) for p in parts] == [
        ("Pay Slip", 2, 3), ("Bank Statement", 4, 4)], parts

    # the rules on the Whitfield sample's facts (samples/synthetic_borrower)
    def F(v, raw, p, src="llm"):
        return {"value": v, "raw": raw, "page": p, "source": src}
    docs = [
        {"doc_type": "Loan Application", "fields": {
            "stated_monthly_income": F(9500.0, "$9,500.00", 1), "employer": F("Sierra Ridge Logistics, Inc.", "Sierra Ridge Logistics, Inc.", 1),
            "checking_balance": F(52400.0, "$52,400.00", 1), "loan_amount": F(420000.0, "$420,000.00", 2),
            "purchase_price": F(525000.0, "$525,000.00", 2), "undisclosed_funds": F(False, "NO", 2)},
         "lists": {"liabilities": [{"creditor": "Capitol Auto Finance", "monthly_payment": "$486.00", "monthly_payment_value": 486.0, "page": 1},
                                   {"creditor": "Chase Card Services", "monthly_payment": "$85.00", "monthly_payment_value": 85.0, "page": 1}]}},
        {"doc_type": "Pay Slip", "fields": {
            "employer": F("Sierra Ridge Logistics, Inc.", "Sierra Ridge Logistics, Inc.", 4), "gross_pay": F(4375.0, "$4,375.00", 4),
            "net_pay": F(2845.31, "$2,845.31", 4), "pay_frequency": F("Semi-monthly", "Semi-monthly", 4)}, "lists": {}},
        {"doc_type": "Bank Statement", "fields": {"ending_balance": F(52489.98, "$52,489.98", 5)}, "lists": {
            "deposits": [{"date": "07/12", "description": "TRANSFER IN FROM ACCT ***9902", "amount": "+$12,000.00", "amount_value": 12000.0, "page": 5},
                         {"date": "07/15", "description": "DIRECT DEP SIERRA RIDGE LOGISTICS PAYROLL", "amount": "+$2,845.31", "amount_value": 2845.31, "page": 5}],
            "debits": [{"date": "07/08", "description": "CAPITOL AUTO FINANCE", "amount": "-$486.00", "amount_value": -486.0, "page": 5},
                       {"date": "07/18", "description": "ONLINE PMT CHASE CARD SERVICES", "amount": "-$620.00", "amount_value": -620.0, "page": 5}]}},
        {"doc_type": "Lender Fee Sheet", "fields": {"loan_amount": F(420000.0, "$420,000.00", 6), "sale_price": F(525000.0, "$525,000.00", 6)}, "lists": {}},
        {"doc_type": "Preliminary Title Report", "fields": {}, "lists": {
            "liens": [{"amount": "$312,000.00", "holder": "Wells Fargo Bank, N.A.", "payoff_required": True, "page": 8}],
            "easements": [{"holder": "Sacramento Municipal Utility District", "purpose": "public utilities", "page": 8}]}},
    ]
    got = {c["id"]: c["status"] for c in run_rules(docs)}
    want = {"income": "mismatch", "employer": "match", "balance": "match", "payroll": "match",
            "large_deposits": "review", "loan_amount": "match", "price": "match",
            "liabilities": "match", "lien": "review", "easement": "info"}
    assert got == want, got
    inc = next(c for c in run_rules(docs) if c["id"] == "income")
    assert "$8,750.00" in inc["detail"] and "overstated by $750.00" in inc["detail"], inc["detail"]

    got = {c["id"]: c["status"] for c in run_rules(docs[:1])}
    assert got["missing_pay_slip"] == "missing" and got["missing_bank_statement"] == "missing"
    table_page = [(5, _flat("07/15 | DIRECT DEP SIERRA RIDGE LOGISTICS PAYROLL | +$2,845.31 | $50,479.00"))]
    assert _locate("07/15 DIRECT DEP SIERRA RIDGE LOGISTICS PAYROLL +$2,845.31", table_page) == 5
    assert _locate("DIRECT DEP ACME PAYROLL", table_page) is None

    class _Blk:
        def __init__(self, content):
            self.content, self.page_num, self.kind = content, 4, "table"

    class _Doc:
        blocks = [_Blk("07/15 | DIRECT DEP SIERRA RIDGE LOGISTICS PAYROLL | +$2,845.31 | $50,479.00")]
        page_start, text = 4, ""

    wrapped = json.dumps({"fields": {}, "lists": {"deposits": [
        {"date": {"value": "07/15"}, "description": {"value": "DIRECT DEP SIERRA RIDGE LOGISTICS PAYROLL"},
         "amount": {"value": "+$2,845.31", "page": 5}}], "debits": []}})
    _f, _l, _d = fields_from_llm("Bank Statement", _Doc(), {}, complete=lambda p: wrapped)
    assert _d == 0 and _l["deposits"][0]["amount_value"] == 2845.31 and _l["deposits"][0]["page"] == 5, _l
    print("review selftest OK")
