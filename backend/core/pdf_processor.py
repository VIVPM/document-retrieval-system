"""
pdf_processor.py — PDF extraction and multi-document analysis pipeline.

Dispatches to one of two extractors based on `EXTRACT_METHOD`:
  textract (default) — AWS Textract, form/table-aware, best on mortgage forms
  pymupdf            — local text-layer read, no AI, no external call

Both return the same page->blocks shape. The LLM classifier then splits
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

PAGE_CONCURRENCY = int(os.getenv("PAGE_CONCURRENCY", "8"))


def _order_blocks(items, y_tol=6.0):
    """Reading order by visual ROW, not by y alone.

    A two-column key-value form (label left, value right) puts the value on the
    same row as its label but a fraction apart in y — so a pure y-sort drops
    every value into one group and every label into another, divorcing them
    ("Interest Rate:" ends up nowhere near "4.250 %"). Group items whose y is
    within y_tol into a row, order each row left-to-right by x, then emit rows
    top-to-bottom.
    """
    clean = [it for it in items if it.get("content", "").strip()]
    clean.sort(key=lambda it: it.get("y_pos", 0))
    rows, ordered = [], []
    for it in clean:
        if rows and abs(it.get("y_pos", 0) - rows[-1]["y"]) <= y_tol:
            rows[-1]["cells"].append(it)
        else:
            rows.append({"y": it.get("y_pos", 0), "cells": [it]})
    for r in rows:
        ordered.extend(sorted(r["cells"], key=lambda it: it.get("x_pos", 0)))
    return ordered


def _extract_textract(file_path: str) -> dict:
    """AWS Textract — form/table-specialised extraction.

    Rasterises each page and calls AnalyzeDocument with TABLES + FORMS. Returns
    {page_no(str): [ {type, y_pos, x_pos, content} ]}. Requires
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (+ optional AWS_REGION).
    """
    import boto3
    import fitz

    sess = boto3.Session(region_name=os.getenv("AWS_REGION", "us-east-1"))
    tex = sess.client("textract")
    doc = fitz.open(file_path)
    print(f"☁️ Textract: {doc.page_count} pages (TABLES + FORMS)...")

    pages = {}
    for pno in range(doc.page_count):
        img = doc[pno].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72)).tobytes("png")
        r = tex.analyze_document(Document={"Bytes": img}, FeatureTypes=["TABLES", "FORMS"])
        B = {b["Id"]: b for b in r["Blocks"]}

        def child_words(blk):
            """Concatenate WORD children (and checked SELECTION_ELEMENTs) of a block."""
            out = []
            for rel in blk.get("Relationships", []) or []:
                if rel["Type"] == "CHILD":
                    for i in rel["Ids"]:
                        b = B[i]
                        if b["BlockType"] == "WORD":
                            out.append(b["Text"])
                        elif b["BlockType"] == "SELECTION_ELEMENT" and b.get("SelectionStatus") == "SELECTED":
                            out.append("[X]")
            return " ".join(out)

        blocks = []
        in_table_word_ids = set()
        for b in r["Blocks"]:
            if b["BlockType"] != "TABLE":
                continue
            cells, mr, mc = {}, 0, 0
            for rel in b.get("Relationships", []) or []:
                if rel["Type"] == "CHILD":
                    for i in rel["Ids"]:
                        c = B[i]
                        if c["BlockType"] == "CELL":
                            cells[(c["RowIndex"], c["ColumnIndex"])] = child_words(c)
                            mr = max(mr, c["RowIndex"])
                            mc = max(mc, c["ColumnIndex"])
                            for r2 in c.get("Relationships", []) or []:
                                if r2["Type"] == "CHILD":
                                    in_table_word_ids.update(r2["Ids"])
            rows = "\n".join(" | ".join(cells.get((ri, ci), "") for ci in range(1, mc + 1))
                             for ri in range(1, mr + 1))
            bb = b.get("Geometry", {}).get("BoundingBox", {}) or {}
            blocks.append({"type": "table", "y_pos": float(bb.get("Top", 0)),
                           "x_pos": float(bb.get("Left", 0)), "content": rows.strip()})

        for b in r["Blocks"]:
            if b["BlockType"] != "LINE":
                continue
            word_ids = []
            for rel in b.get("Relationships", []) or []:
                if rel["Type"] == "CHILD":
                    word_ids.extend(rel["Ids"])
            if word_ids and all(w in in_table_word_ids for w in word_ids):
                continue
            bb = b.get("Geometry", {}).get("BoundingBox", {}) or {}
            blocks.append({"type": "text", "y_pos": float(bb.get("Top", 0)),
                           "x_pos": float(bb.get("Left", 0)),
                           "content": (b.get("Text") or "").strip()})

        pages[str(pno + 1)] = [b for b in blocks if b["content"]]

    doc.close()
    print(f"✅ Textract extraction complete! {len(pages)} pages.")
    return pages


def _extract_pymupdf(file_path: str, y_tol: float = 3.0) -> dict:
    """Local, NO-AI extraction (PyMuPDF).

    Reads the PDF text layer directly and groups words into visual lines by
    (y, x) — producing the same page->blocks shape Textract emits, so the
    downstream pipeline is agnostic to which extractor ran. No GPU, no API,
    milliseconds not seconds. Caveat: text-layer only — a SCANNED page yields
    nothing (Textract's OCR is what covers that case).
    """
    import fitz

    doc = fitz.open(file_path)
    pages, scanned = {}, []
    try:
        for pno in range(doc.page_count):
            words = doc[pno].get_text("words")
            if not words:
                scanned.append(pno + 1)
            words.sort(key=lambda w: (w[1], w[0]))
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
    Extract text from a PDF with Textract and detect logical document boundaries
    using the LLM classifier.

    Tables are converted to pipe-delimited text rows and merged into the page
    text in reading order (top-to-bottom by Y position).

    Args:
        pdf_file: file path string or file-like object.

    Returns:
        pages_info   : one PageInfo per PDF page (0-indexed)
        logical_docs : detected logical documents with combined text
    """
    if on_stage:
        on_stage("extract")

    import tempfile

    file_path = pdf_file
    is_temp = False
    if not isinstance(pdf_file, str):
        file_path = getattr(pdf_file, 'name', None)
        if not file_path:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                file_path = tmp.name
                is_temp = True

    method = os.getenv("EXTRACT_METHOD", "textract").lower()
    print(f"📖 Starting PDF extraction (EXTRACT_METHOD={method})...")

    try:
        if method == "pymupdf":
            pages_data = _extract_pymupdf(file_path)
        elif method == "textract":
            pages_data = _extract_textract(file_path)
        else:
            print(f"⚠️ Unknown EXTRACT_METHOD={method!r}; falling back to textract.")
            pages_data = _extract_textract(file_path)

        pages_info: List[PageInfo] = []

        for str_page_no in sorted(pages_data.keys(), key=int):
            page_no = int(str_page_no)
            items = _order_blocks(pages_data[str_page_no])

            blocks = [
                Block(
                    kind=item.get("type", "text"),
                    content=item["content"],
                    page_num=page_no - 1,
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

    if on_stage:
        on_stage("split")
    print("🧠 Analysing document structure...")

    with ThreadPoolExecutor(max_workers=PAGE_CONCURRENCY) as pool:
        page_types = list(pool.map(
            lambda p: classify_document_type(p.text), pages_info))

        same_as_prev = list(pool.map(
            lambda i: detect_document_boundary(
                pages_info[i - 1].text, pages_info[i].text, page_types[i - 1]),
            range(1, len(pages_info)),
        ))

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
