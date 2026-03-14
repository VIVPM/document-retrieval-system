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

from typing import List, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from core.models import PageInfo, LogicalDocument
from core.document_classifier import classify_document_type, detect_document_boundary


def extract_and_analyze_pdf(
    pdf_file,
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

    print("☁️ Sending PDF to Modal Cloud for Docling Extractions (OCR + Tables)...")

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
            response = requests.post(MODAL_DOCLING_URL, files=files, timeout=600)
            
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
            page_text = "\n\n".join(item["content"] for item in items)
            
            pages_info.append(PageInfo(
                page_num=page_no - 1,   # 0-indexed for downstream
                text=page_text,
            ))
            print(f"  Page {page_no}: {len(page_text)} chars")

        if not pages_info:
            raise ValueError("No text could be extracted from PDF")

    finally:
        if is_temp and os.path.exists(file_path):
            os.remove(file_path)

    # ------------------------------------------------------------------
    # Document boundary detection → build LogicalDocument list
    # ------------------------------------------------------------------
    print("🧠 Analysing document structure...")
    logical_docs: List[LogicalDocument] = []
    current_doc_type: str = None
    current_doc_pages: List[PageInfo] = []
    doc_counter = 0

    for i, page_info in enumerate(pages_info):
        if i == 0:
            current_doc_type = classify_document_type(page_info.text)
            page_info.doc_type = current_doc_type
            page_info.page_in_doc = 0
            current_doc_pages = [page_info]
            print(f"  Page {i}: New document detected — {current_doc_type}")
        else:
            prev_text = pages_info[i - 1].text
            is_same = detect_document_boundary(prev_text, page_info.text, current_doc_type)

            if is_same:
                page_info.doc_type = current_doc_type
                page_info.page_in_doc = len(current_doc_pages)
                current_doc_pages.append(page_info)
            else:
                # Save completed logical document
                logical_docs.append(LogicalDocument(
                    doc_id=f"doc_{doc_counter}",
                    doc_type=current_doc_type,
                    page_start=current_doc_pages[0].page_num,
                    page_end=current_doc_pages[-1].page_num,
                    text="\n\n".join(p.text for p in current_doc_pages),
                ))
                doc_counter += 1

                # Start new logical document
                current_doc_type = classify_document_type(page_info.text)
                page_info.doc_type = current_doc_type
                page_info.page_in_doc = 0
                current_doc_pages = [page_info]
                print(f"  Page {i}: New document detected — {current_doc_type}")

    # Flush last document
    if current_doc_pages:
        logical_docs.append(LogicalDocument(
            doc_id=f"doc_{doc_counter}",
            doc_type=current_doc_type,
            page_start=current_doc_pages[0].page_num,
            page_end=current_doc_pages[-1].page_num,
            text="\n\n".join(p.text for p in current_doc_pages),
        ))

    print(f"✅ Identified {len(logical_docs)} logical documents")
    for ld in logical_docs:
        print(f"   - {ld.doc_type}: Pages {ld.page_start}–{ld.page_end}")

    return pages_info, logical_docs
