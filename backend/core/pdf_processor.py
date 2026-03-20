"""
pdf_processor.py — PDF extraction and multi-document analysis pipeline.

Uses Docling for high-quality PDF extraction (text + tables) and then
applies LLM-based document classification and boundary detection to split
a multi-document PDF into individual logical documents.

Public API
----------
extract_and_analyze_pdf(pdf_file)
    → Tuple[List[PageInfo], List[LogicalDocument]]
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from core.models import Block, PageInfo, LogicalDocument
from core.document_classifier import classify_document_type, detect_document_boundary

# Concurrent per-page LLM calls during structure analysis.
PAGE_CONCURRENCY = int(os.getenv("PAGE_CONCURRENCY", "8"))


def _order_blocks(items, y_tol=6.0):
    """Reading order by visual ROW, not by y alone.

    A two-column key-value form (label left, value right) puts the value on the
    same row as its label but a fraction apart in y — so a pure y-sort drops
    every value into one group and every label into another, divorcing them
    ("Interest Rate:" ends up nowhere near "4.250 %"). Group items whose y is
    within y_tol into a row, order each row left-to-right by x, then emit rows
    top-to-bottom. Falls back cleanly if x is absent (an older worker that
    emitted only y): every cell then sorts to x=0 and the order is the old
    y-order.
    """
    clean = [it for it in items if it.get("content", "").strip()]
    clean.sort(key=lambda it: it.get("y_pos", 0))   # keep the existing page order
    rows, ordered = [], []
    for it in clean:
        if rows and abs(it.get("y_pos", 0) - rows[-1]["y"]) <= y_tol:
            rows[-1]["cells"].append(it)
        else:
            rows.append({"y": it.get("y_pos", 0), "cells": [it]})
    for r in rows:
        ordered.extend(sorted(r["cells"], key=lambda it: it.get("x_pos", 0)))
    return ordered


def _extract_docling(file_path: str) -> dict:
    """Docling on Modal — AI layout + table models, includes OCR (handles
    scanned pages). Returns {page_no(str): [ {type, y_pos, x_pos, content} ]}."""
    import os
    import requests

    url = os.getenv("DOCLING_URL")
    if not url:
        raise EnvironmentError(
            "DOCLING_URL is not set. Add it to your .env file.\n"
            "  DOCLING_URL=https://<your-deployment>.modal.run/extract"
        )
    # "classic" = layout model + TableFormer, "vlm" = granite-docling-258M.
    pipeline = os.getenv("DOCLING_PIPELINE", "classic")
    print(f"☁️ Sending PDF to Modal for Docling (pipeline={pipeline}, OCR + tables)...")
    with open(file_path, "rb") as f:
        resp = requests.post(
            url, files={"file": (os.path.basename(file_path), f, "application/pdf")},
            params={"pipeline": pipeline},
            timeout=1800 if pipeline == "vlm" else 600,
        )
    if resp.status_code != 200:
        raise ValueError(f"Modal API Error: {resp.text}")
    result = resp.json()
    if not result.get("success"):
        raise ValueError(f"Docling Extraction Failed: {result.get('error')}")
    print(f"✅ Cloud extraction complete! Found {result['num_pages']} pages.")
    return result["pages"]


def _extract_pymupdf(file_path: str, y_tol: float = 3.0) -> dict:
    """Local, NO-AI extraction (PyMuPDF). Reads the PDF text layer directly and
    groups words into visual lines by (y, x) — the same reassembly the Modal
    path relies on — producing the identical page->blocks shape. No GPU, no
    Modal, milliseconds not seconds. Caveat: text-layer only, so a SCANNED
    page yields nothing (that is the one thing Docling's OCR adds)."""
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)
    pages, scanned = {}, []
    try:
        for pno in range(doc.page_count):
            words = doc[pno].get_text("words")  # (x0, y0, x1, y1, word, blk, ln, wno)
            if not words:
                scanned.append(pno + 1)
            words.sort(key=lambda w: (w[1], w[0]))   # y then x
            blocks, line = [], None
            for w in words:
                if line is not None and abs(w[1] - line["y"]) <= y_tol:
                    line["cells"].append(w)
                else:
                    line = {"y": w[1], "cells": [w]}
                    blocks.append(line)
            page_blocks = []
            for line in blocks:
                cells = sorted(line["cells"], key=lambda w: w[0])
                text = " ".join(c[4] for c in cells).strip()
                if text:
                    page_blocks.append({"type": "text", "y_pos": float(line["y"]),
                                        "x_pos": float(cells[0][0]), "content": text})
            pages[str(pno + 1)] = page_blocks
    finally:
        doc.close()
    note = f"; ⚠️ {len(scanned)} scanned page(s) with no text layer: {scanned}" if scanned else ""
    print(f"✅ PyMuPDF extraction complete! {len(pages)} pages{note}")
    return pages


def extract_and_analyze_pdf(
    pdf_file,
    filename: str = "document.pdf",
    on_stage=None,
) -> Tuple[List[PageInfo], List[LogicalDocument]]:
    """
    Extract text from a PDF with Docling and detect logical document
    boundaries using the LLM classifier.

    Tables are converted to pipe-delimited text rows and merged into the
    page text in reading order (top-to-bottom by Y position).

    Args:
        pdf_file: file path string or file-like object accepted by Docling.

    Returns:
        pages_info   : one PageInfo per PDF page (0-indexed)
        logical_docs : detected logical documents with combined text
    """
    if on_stage:
        on_stage("extract")

    import tempfile
    import os

    # docling = Modal AI pipeline (layout/table models + OCR); pymupdf = local,
    # no-AI, text-layer only. Default docling. See _extract_* above.
    method = os.getenv("EXTRACT_METHOD", "docling").lower()
    print(f"📖 Starting PDF extraction (EXTRACT_METHOD={method})...")

    file_path = pdf_file
    is_temp = False
    if not isinstance(pdf_file, str):
        file_path = getattr(pdf_file, 'name', None)
        if not file_path:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                file_path = tmp.name
                is_temp = True

    try:
        pages_data = (_extract_pymupdf(file_path) if method == "pymupdf"
                      else _extract_docling(file_path))

        pages_info: List[PageInfo] = []

        for str_page_no in sorted(pages_data.keys(), key=int):
            page_no = int(str_page_no)
            items = _order_blocks(pages_data[str_page_no])

            # Keep Docling's type tag; the chunker needs it to keep tables whole.
            blocks = [
                Block(
                    kind=item.get("type", "text"),
                    content=item["content"],
                    page_num=page_no - 1,        # 0-indexed for downstream
                )
                for item in items
                if item.get("content", "").strip()
            ]
            page_text = "\n\n".join(b.content for b in blocks)

            pages_info.append(PageInfo(
                page_num=page_no - 1,
                text=page_text,
                blocks=blocks,
            ))
            n_tables = sum(1 for b in blocks if b.kind == "table")
            print(f"  Page {page_no}: {len(page_text)} chars, "
                  f"{len(blocks)} blocks ({n_tables} tables)")

        if not pages_info:
            raise ValueError("No text could be extracted from PDF")

    finally:
        if is_temp and os.path.exists(file_path):
            os.remove(file_path)

    # ------------------------------------------------------------------
    # Document boundary detection → build LogicalDocument list
    # ------------------------------------------------------------------
    if on_stage:
        on_stage("split")
    print("🧠 Analysing document structure...")

    # The sequential version fed each boundary check the running document
    # type, which is what made it sequential. Every page is classified up
    # front instead, so the check can use the PREVIOUS page's own type and
    # every pair becomes independent — two concurrent rounds instead of one
    # round-trip per page.
    #
    # Dropping the hint entirely was tried first and is wrong: without it the
    # Contract following a Pay Slip is not detected and the two merge.
    with ThreadPoolExecutor(max_workers=PAGE_CONCURRENCY) as pool:
        page_types = list(pool.map(
            lambda p: classify_document_type(p.text), pages_info))

        same_as_prev = list(pool.map(
            lambda i: detect_document_boundary(
                pages_info[i - 1].text, pages_info[i].text, page_types[i - 1]),
            range(1, len(pages_info)),
        ))

    # A new logical document starts wherever the boundary check says "no".
    starts = [0] + [i for i, same in enumerate(same_as_prev, start=1) if not same]
    doc_types = [page_types[i] for i in starts]

    logical_docs: List[LogicalDocument] = []
    for doc_counter, (start, doc_type) in enumerate(zip(starts, doc_types)):
        end = starts[doc_counter + 1] if doc_counter + 1 < len(starts) else len(pages_info)
        doc_pages = pages_info[start:end]
        for offset, page_info in enumerate(doc_pages):
            page_info.doc_type = doc_type
            page_info.page_in_doc = offset
        print(f"  Page {start}: New document detected — {doc_type}")

        logical_docs.append(LogicalDocument(
            doc_id=f"doc_{doc_counter}",
            doc_type=doc_type,
            page_start=doc_pages[0].page_num,
            page_end=doc_pages[-1].page_num,
            text="\n\n".join(p.text for p in doc_pages),
            filename=filename,
            blocks=[b for p in doc_pages for b in p.blocks],
        ))

    print(f"✅ Identified {len(logical_docs)} logical documents")
    for ld in logical_docs:
        print(f"   - {ld.doc_type}: Pages {ld.page_start}–{ld.page_end}")

    return pages_info, logical_docs
