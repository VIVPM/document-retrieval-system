"""A/B: classic vs VLM vs a HYBRID (classic tables + VLM text) on one PDF.

The roadmap names extraction as the answer-quality ceiling: classic keeps the
fee tables but detaches text fields (Interest Rate: 4.250%); VLM fixes the text
fields but empties tables. The hybrid takes each pipeline's win — VLM's text
blocks + classic's table blocks per page — merged client-side, so it can be
measured WITHOUT deploying a new Modal pipeline.

    EVAL_PDF="Blob File Sample.pdf" PYTHONIOENCODING=utf-8 \
        python backend/eval/hybrid_extract_eval.py

Caches each pipeline as extract_<name>.json; delete to re-fetch.
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

# Fields where classic is known to detach the value from its label, plus
# controls it gets right. Not all appear in every document.
PROBES = ["ORIGINATION", "Total Loan", "Interest Rate", "Funds needed to close",
          "Gross", "Net Pay", "Loan Amount"]


def run(pipeline: str) -> dict:
    cache = os.path.join(os.path.dirname(__file__), f"extract_{pipeline}.json")
    if os.path.exists(cache):
        print(f"[{pipeline}] cached")
        return json.load(open(cache, encoding="utf-8"))
    print(f"[{pipeline}] posting {os.path.basename(PDF)} ...")
    t0 = time.time()
    with open(PDF, "rb") as f:
        r = requests.post(URL, files={"file": (os.path.basename(PDF), f, "application/pdf")},
                          params={"pipeline": pipeline}, timeout=1800)
    r.raise_for_status()
    data = r.json()
    data["_secs"] = round(time.time() - t0, 1)
    json.dump(data, open(cache, "w", encoding="utf-8"), indent=2)
    print(f"   {data['_secs']}s success={data.get('success')}")
    return data


def make_hybrid(classic: dict, vlm: dict) -> dict:
    """VLM text blocks + classic table blocks, per page. Falls back to classic
    text where VLM produced none."""
    cp, vp = classic.get("pages") or {}, vlm.get("pages") or {}
    pages = {}
    for pg in sorted(set(cp) | set(vp), key=int):
        c_blocks, v_blocks = cp.get(pg, []), vp.get(pg, [])
        c_tables = [b for b in c_blocks if b.get("type") == "table"]
        v_text = [b for b in v_blocks if b.get("type") != "table"]
        if not v_text:                      # VLM gave no text for this page
            v_text = [b for b in c_blocks if b.get("type") != "table"]
        pages[pg] = v_text + c_tables
    return {"pages": pages, "num_pages": classic.get("num_pages", 0),
            "_secs": round(classic.get("_secs", 0) + vlm.get("_secs", 0), 1)}


def blocks(data):
    for page, items in sorted((data.get("pages") or {}).items(), key=lambda t: int(t[0])):
        for b in items:
            yield int(page), b


def summarise(name, data):
    bs = list(blocks(data))
    tables = [b for _, b in bs if b.get("type") == "table"]
    empty_tables = [b for b in tables if not b.get("content", "").strip()]
    chars = sum(len(b.get("content", "")) for _, b in bs)
    print(f"  {name:8} {len(bs):>4} blocks  {len(tables):>3} tables "
          f"({len(empty_tables)} empty)  {chars:>7} chars  {data.get('_secs','?')}s")
    return bs


def probe(bs, needle):
    """Rows mentioning `needle`; count those with a digit on the same row —
    the shape that means the value stayed attached to its label."""
    hits = with_num = 0
    for _, b in bs:
        for row in b.get("content", "").split("\n"):
            if needle.lower() in row.lower():
                hits += 1
                if any(c.isdigit() for c in row):
                    with_num += 1
    return hits, with_num


if not URL:
    sys.exit("DOCLING_URL not set")

print(f"PDF: {os.path.basename(PDF)}\n")
classic, vlm = run("classic"), run("vlm")
hybrid = make_hybrid(classic, vlm)
arms = {"classic": classic, "vlm": vlm, "hybrid": hybrid}

print("\n" + "=" * 70 + "\nTOTALS\n")
bs = {name: summarise(name, d) for name, d in arms.items()}

print("\n" + "=" * 70 + "\nPROBES — rows with the label, and how many keep a number on the same row\n")
print(f"  {'field':22} {'classic':>16} {'vlm':>16} {'hybrid':>16}")
for needle in PROBES:
    cells = []
    for name in ("classic", "vlm", "hybrid"):
        h, n = probe(bs[name], needle)
        cells.append(f"{n}/{h}" if h else "—")
    print(f"  {needle:22} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")
print("\n(x/y = y rows mention the label, x of them keep a number on the same row)")
