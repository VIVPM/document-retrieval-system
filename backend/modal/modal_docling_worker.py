import base64
import os
import tempfile
import modal
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Define the Docling image with necessary system dependencies (like Tesseract for OCR)
docling_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("tesseract-ocr", "libtesseract-dev", "poppler-utils")
    .pip_install("docling==2.11.0", "fastapi[standard]", "python-multipart", "docling-ibm-models")
)

app = modal.App("docling-pdf-extractor")
web_app = FastAPI(title="Docling Extraction API")

class DoclingResponse(BaseModel):
    success: bool
    num_pages: int
    pages: Dict[int, List[Dict[str, Any]]]
    error: str = None

@app.function(
    image=docling_image,
    gpu="T4",
    memory=4096, 
    timeout=600, # 10 mins max per PDF
)
@modal.asgi_app()
def serve_docling():
    """FastAPI server to receive PDFs and process them with Docling."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    
    @web_app.post("/extract", response_model=DoclingResponse)
    async def extract_pdf(file: UploadFile = File(...)):
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
            
        # 1. Save uploaded file to a temporary location
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
                
            print(f"Processing uploaded PDF: {file.filename} ({len(content)} bytes)")
                
            # 2. Configure Docling with ALL features enabled (since we have cloud memory!)
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = True
            pipeline_options.do_ocr = True # We can safely turn OCR back on in the cloud
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            # 3. Process the document
            result = converter.convert(tmp_path)
            doc = result.document
            
            # 4. Extract data into our expected structure
            page_content = {page_no: [] for page_no in doc.pages.keys()}

            for text_item in doc.texts:
                if text_item.prov and len(text_item.prov) > 0:
                    prov = text_item.prov[0]
                    page_no = prov.page_no
                    y_pos = prov.bbox.t if prov.bbox else 0
                    page_content[page_no].append({
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
                    page_content[page_no].append({
                        "type": "table",
                        "y_pos": y_pos,
                        "content": "\n".join(table_lines),
                    })
                    
            return DoclingResponse(
                success=True,
                num_pages=len(doc.pages),
                pages=page_content
            )
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return DoclingResponse(success=False, num_pages=0, pages={}, error=str(e))
            
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    return web_app
