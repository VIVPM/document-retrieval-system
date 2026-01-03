"""
Docling PDF extraction on Modal.

Two pipelines behind one endpoint, chosen per request with `?pipeline=`:
classic (layout model + TableFormer, the default) and vlm (granite-docling-258M).
See upgrade_roadmap.txt item 24 for the comparison.

L4 rather than T4 because granite-docling needs bfloat16.

Deploy:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal deploy modal/modal_docling_worker.py
"""
import os
import tempfile
from typing import Any, Dict, List

import modal
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

GRANITE_REPO = "ibm-granite/granite-docling-258M"

# docling unpinned: Modal content-addresses the built image, so a deployed
# worker stays fixed until this definition changes.
docling_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "tesseract-ocr", "libtesseract-dev", "poppler-utils",
        # opencv is a transitive dep of docling >=2.117 and needs these, or
        # the image deploys clean and every request dies on libGL.so.1.
        "libgl1", "libglib2.0-0",
    )
    .pip_install(
        "docling", "docling-ibm-models",
        "transformers", "accelerate",         # required by the VLM pipeline
        "fastapi[standard]", "python-multipart",
    )
    # Bake the weights in; downloading ~500MB on each cold start is worse.
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{GRANITE_REPO}')\""
    )
)

app = modal.App("docling-pdf-extractor")
web_app = FastAPI(title="Docling Extraction API")


class DoclingResponse(BaseModel):
    success: bool
    num_pages: int
    pages: Dict[int, List[Dict[str, Any]]]
    pipeline: str = "classic"
    error: str | None = None


@app.function(
    image=docling_image,
    gpu="L4",             # bfloat16; see module docstring
    memory=8192,          # the VLM holds page images and weights at once
    timeout=1800,         # 30 min: the VLM is per-page, so long packets are slow
)
@modal.asgi_app()
def serve_docling():
    """FastAPI server to receive PDFs and process them with Docling."""
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (PdfPipelineOptions,
                                                    VlmPipelineOptions)
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    # Building a converter loads models, so keep one per pipeline per container.
    _converters: Dict[str, DocumentConverter] = {}

    def converter_for(pipeline: str) -> DocumentConverter:
        if pipeline not in _converters:
            if pipeline == "vlm":
                opts = VlmPipelineOptions(
                    vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
                    generate_page_images=True,
                )
                fmt = PdfFormatOption(pipeline_cls=VlmPipeline,
                                      pipeline_options=opts)
            else:
                opts = PdfPipelineOptions()
                opts.do_table_structure = True
                opts.do_ocr = True
                fmt = PdfFormatOption(pipeline_options=opts)
            print(f"Building '{pipeline}' converter...")
            _converters[pipeline] = DocumentConverter(
                format_options={InputFormat.PDF: fmt})
        return _converters[pipeline]

    @web_app.post("/extract", response_model=DoclingResponse)
    async def extract_pdf(
        file: UploadFile = File(...),
        pipeline: str = Query("classic", pattern="^(classic|vlm)$"),
    ):
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            print(f"Processing {file.filename} ({len(content)} bytes) "
                  f"with the '{pipeline}' pipeline")

            result = converter_for(pipeline).convert(tmp_path)
            doc = result.document

            page_content = {page_no: [] for page_no in doc.pages.keys()}

            for text_item in doc.texts:
                if text_item.prov and len(text_item.prov) > 0:
                    prov = text_item.prov[0]
                    page_no = prov.page_no
                    y_pos = prov.bbox.t if prov.bbox else 0
                    # setdefault: the VLM can emit items on unmapped pages.
                    page_content.setdefault(page_no, []).append({
                        "type": "text",
                        "y_pos": y_pos,
                        "content": text_item.text,
                    })

            for table in doc.tables:
                if table.prov and len(table.prov) > 0:
                    prov = table.prov[0]
                    page_no = prov.page_no
                    y_pos = prov.bbox.t if prov.bbox else 0
                    table_lines = [
                        " | ".join(cell.text for cell in row)
                        for row in table.data.grid
                    ]
                    page_content.setdefault(page_no, []).append({
                        "type": "table",
                        "y_pos": y_pos,
                        "content": "\n".join(table_lines),
                    })

            return DoclingResponse(
                success=True,
                num_pages=len(doc.pages),
                pages=page_content,
                pipeline=pipeline,
            )

        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return DoclingResponse(success=False, num_pages=0, pages={},
                                   pipeline=pipeline, error=str(e))

        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return web_app
