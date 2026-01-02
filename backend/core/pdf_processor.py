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


def extract_and_analyze_pdf(
    pdf_file,
    filename: str = "document.pdf",
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
    print("📖 Starting PDF extraction with Docling...")

    import tempfile
    import requests
    import os

    MODAL_DOCLING_URL = os.getenv("DOCLING_URL")
    if not MODAL_DOCLING_URL:
        raise EnvironmentError(
            "DOCLING_URL is not set. Add it to your .env file.\n"
            "  DOCLING_URL=https://<your-deployment>.modal.run/extract"
        )

    # "classic" = layout model + TableFormer, "vlm" = granite-docling-258M.
    DOCLING_PIPELINE = os.getenv("DOCLING_PIPELINE", "classic")

    print(f"☁️ Sending PDF to Modal Cloud for Docling Extractions "
          f"(pipeline={DOCLING_PIPELINE}, OCR + Tables)...")

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
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/pdf")}
            # The VLM runs the model once per page, so it needs far longer.
            response = requests.post(
                MODAL_DOCLING_URL, files=files,
                params={"pipeline": DOCLING_PIPELINE},
                timeout=1800 if DOCLING_PIPELINE == "vlm" else 600,
            )
            
        if response.status_code != 200:
            raise ValueError(f"Modal API Error: {response.text}")
            
        result = response.json()
        
        if not result.get("success"):
            raise ValueError(f"Docling Extraction Failed: {result.get('error')}")
            
        print(f"✅ Cloud extraction complete! Found {result['num_pages']} pages.")
        
        pages_info: List[PageInfo] = []
        pages_data = result["pages"]
        
        for str_page_no in sorted(pages_data.keys(), key=int):
            page_no = int(str_page_no)
            items = sorted(pages_data[str_page_no], key=lambda x: x["y_pos"])

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
