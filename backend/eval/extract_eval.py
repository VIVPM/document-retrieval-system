"""
Compares the classic and vlm Docling pipelines on the same PDF.

Prints totals plus the rows around each probe field, so a pipeline that wins a
probe while dropping content elsewhere is visible. Caches each response as
extract_<pipeline>.json; delete to re-fetch.
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from dotenv import load_dotenv

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BACKEND)
PDF = os.getenv("EVAL_PDF", os.path.join(REPO, "Test Blob File.pdf"))
load_dotenv(os.path.join(BACKEND, ".env"))
URL = os.getenv("DOCLING_URL")

# Fields damaged by the classic pipeline, plus controls it already gets right.
PROBES = [
    "FUNDS NEEDED TO CLOSE",     # the known break
    "ORIGINATION",
    "Total Loan Costs",
    "Interest Rate",
    "Gross Earnings",
    "Net Pay",
]


def run(pipeline: str) -> dict:
    cache = f"extract_{pipeline}.json"
    if os.path.exists(cache):
        print(f"[{pipeline}] using cached {cache}")
        return json.load(open(cache, encoding="utf-8"))

    print(f"[{pipeline}] posting {os.path.basename(PDF)} ...")
    t0 = time.time()
    with open(PDF, "rb") as f:
        r = requests.post(
            URL, files={"file": (os.path.basename(PDF), f, "application/pdf")},
            params={"pipeline": pipeline}, timeout=1800,
        )
    r.raise_for_status()
    data = r.json()
    data["_secs"] = round(time.time() - t0, 1)
    if not data.get("success"):
        print(f"   FAILED: {data.get('error')}")
    json.dump(data, open(cache, "w", encoding="utf-8"), indent=2)
    print(f"   {data['_secs']}s, success={data.get('success')}, "
          f"pipeline_reported={data.get('pipeline')}")
    return data


def blocks(data):
    for page, items in sorted((data.get("pages") or {}).items(), key=lambda t: int(t[0])):
        for b in items:
            yield int(page), b


def summarise(name, data):
    bs = list(blocks(data))
    tables = [b for _, b in bs if b["type"] == "table"]
    chars = sum(len(b["content"]) for _, b in bs)
    print(f"  {name:9} {data.get('num_pages',0):>3} pages  {len(bs):>4} blocks  "
          f"{len(tables):>3} tables  {chars:>7} chars  {data.get('_secs','?')}s")
    return bs


def probe(name, bs, needle):
    """Print every row mentioning `needle`, with the row after it."""
    found = []
    for page, b in bs:
        rows = b["content"].split("\n")
        for i, row in enumerate(rows):
            if needle.lower() in row.lower():
                nxt = rows[i + 1] if i + 1 < len(rows) else ""
                found.append((page, b["type"], row.strip(), nxt.strip()))
    if not found:
        print(f"    {name:9} — NOT FOUND")
    for page, kind, row, nxt in found[:3]:
        print(f"    {name:9} p{page} [{kind}] {row[:118]}")
        if nxt:
            print(f"    {'':9}      next: {nxt[:112]}")
    return found


if not URL:
    sys.exit("DOCLING_URL not set")

results = {p: run(p) for p in ("classic", "vlm")}

print("\n" + "=" * 78 + "\nTOTALS\n")
bs = {name: summarise(name, d) for name, d in results.items()}

print("\n" + "=" * 78 + "\nPROBE FIELDS — does the value stay attached to its label?\n")
for needle in PROBES:
    print(f"\n  ── {needle}")
    hits = {name: probe(name, b, needle) for name, b in bs.items()}
    for name, h in hits.items():
        # A row carrying a digit alongside the label is the shape we want.
        with_number = sum(1 for _, _, row, _ in h if any(c.isdigit() for c in row))
        print(f"    {name:9} → {len(h)} row(s), {with_number} with a number on the same row")

print("\nRaw output kept in extract_classic.json / extract_vlm.json "
      "(delete to re-fetch).")
