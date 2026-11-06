"""Standalone AWS Textract Forms + Tables extractor."""

import os


def extract_textract(file_path: str) -> dict:
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
