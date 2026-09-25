"""pdf_processor.py — PDF extraction and multi-document analysis pipeline."""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from core.models import Block, PageInfo, LogicalDocument
from core.document_classifier import classify_document_type, detect_document_boundary

PAGE_CONCURRENCY = int(os.getenv("PAGE_CONCURRENCY", "8"))


def _order_blocks(items, y_tol=6.0):
    """Reading order by visual ROW, not by y alone."""
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
    """AWS Textract — form/table-specialised extraction."""
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
            if b["BlockType"] != "KEY_VALUE_SET" or "KEY" not in b.get("EntityTypes", []):
                continue
            key = child_words(b)
            value = " ".join(child_words(B[i]) for rel in b.get("Relationships", []) or []
                             if rel["Type"] == "VALUE" for i in rel["Ids"] if i in B)
            if key.strip() and value.strip():
                blocks.append({"type": "kv", "key": key.strip(), "value": value.strip(),
                               "content": ""})

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

        pages[str(pno + 1)] = [b for b in blocks if b["content"] or b["type"] == "kv"]

    doc.close()
    print(f"✅ Textract extraction complete! {len(pages)} pages.")
    return pages


def _flatten_tables(text: str) -> str:
    """Table cell separators replaced by spaces, for the boundary check on
    PyMuPDF text: with "a | b" rows the model called different documents the same."""
    return re.sub(r"[ \t]*(?:\|[ \t]*)+", " ", text)


def _pymupdf_tables(page) -> list:
    """Tables on a page as {bbox, content}, rows joined with " | " like Textract's."""
    try:
        found = page.find_tables().tables
    except Exception as e:
        print(f"⚠️ Table detection failed on page {page.number + 1}: {e}")
        return []
    out = []
    for t in found:
        rows = [" | ".join((c or "").replace("\n", " ").strip() for c in row)
                for row in t.extract()]
        content = "\n".join(r for r in rows if r.replace("|", "").strip())
        if content:
            out.append({"bbox": tuple(t.bbox), "content": content})
    return out


def _extract_pymupdf(file_path: str, y_tol: float = 3.0) -> dict:
    """Local, NO-AI extraction (PyMuPDF)."""
    import fitz

    doc = fitz.open(file_path)
    pages, scanned = {}, []
    try:
        for pno in range(doc.page_count):
            page = doc[pno]
            words = page.get_text(
                "words", flags=fitz.TEXTFLAGS_WORDS & ~fitz.TEXT_PRESERVE_LIGATURES)
            if not words:
                scanned.append(pno + 1)
            tables = _pymupdf_tables(page) if words else []
            if tables:
                rects = [fitz.Rect(t["bbox"]) for t in tables]
                words = [w for w in words
                         if not any(r.contains(fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))
                                    for r in rects)]
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
            for t in tables:
                page_blocks.append({"type": "table", "y_pos": float(t["bbox"][1]),
                                    "x_pos": float(t["bbox"][0]), "content": t["content"]})
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
    """    Extract text from a PDF with Textract and detect logical document boundaries
    using the LLM classifier."""
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

    y_tol = 6.0 if method == "pymupdf" else 0.005

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
            raw = pages_data[str_page_no]
            kv = [(it["key"], it["value"]) for it in raw if it.get("type") == "kv"]
            items = _order_blocks([it for it in raw if it.get("type") != "kv"], y_tol=y_tol)

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
                kv=kv,
            ))
            n_tables = sum(1 for b in blocks if b.kind == "table")
            print(f"  Page {page_no}: {len(page_text)} chars, "
                  f"{len(blocks)} blocks ({n_tables} tables)")

        if not any(p.text.strip() for p in pages_info):
            raise ValueError(
                "No text could be read from this PDF. It looks scanned; scanned "
                "pages need EXTRACT_METHOD=textract, or upload a PDF with a text layer.")

    finally:
        if is_temp and os.path.exists(file_path):
            os.remove(file_path)

    if on_stage:
        on_stage("split")
    print("🧠 Analysing document structure...")

    boundary_text = _flatten_tables if method == "pymupdf" else (lambda t: t)
    with ThreadPoolExecutor(max_workers=PAGE_CONCURRENCY) as pool:
        page_types = list(pool.map(
            lambda p: classify_document_type(p.text), pages_info))

        same_as_prev = list(pool.map(
            lambda i: detect_document_boundary(
                boundary_text(pages_info[i - 1].text), boundary_text(pages_info[i].text),
                page_types[i - 1]),
            range(1, len(pages_info)),
        ))

    for p, t in zip(pages_info, page_types):
        p.page_type = t
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
            kv=[(k, v, p.page_num) for p in doc_pages for k, v in p.kv],
            page_types=[p.page_type or doc_type for p in doc_pages],
        ))

    print(f"✅ Identified {len(logical_docs)} logical documents")
    for ld in logical_docs:
        print(f"   - {ld.doc_type}: Pages {ld.page_start}–{ld.page_end}")

    return pages_info, logical_docs
