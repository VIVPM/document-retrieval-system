"""Convert Reducto Parse blocks into the PDF processor's page/block format."""

from html.parser import HTMLParser

import requests


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self.parts.append("\n")


class _Table(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def _plain(markup):
    parser = _Text()
    parser.feed(markup or "")
    return " ".join("".join(parser.parts).split())


def pages_from_reducto(data):
    """Keep page citations and turn HTML tables into searchable row text and KVs."""
    count = (data.get("usage") or {}).get("num_pages")
    if not isinstance(count, int) or count < 1:
        raise ValueError("Reducto response has no valid page count")
    pages = {str(page): [] for page in range(1, count + 1)}
    chunks = (data.get("result") or {}).get("chunks") or []
    for chunk in chunks:
        for block in chunk.get("blocks") or []:
            bbox = block.get("bbox") or {}
            page = bbox.get("page")
            if not isinstance(page, int) or not 1 <= page <= count:
                raise ValueError("Reducto block has an invalid page citation")
            raw = block.get("content") or ""
            kind = block.get("type")
            top, left = bbox.get("top", 0), bbox.get("left", 0)
            if kind == "Table":
                parser = _Table()
                parser.feed(raw)
                rows = [row for row in parser.rows if any(row)]
                content = "\n".join(" | ".join(row) for row in rows)
                if content:
                    pages[str(page)].append({"type": "table", "y_pos": top,
                                             "x_pos": left, "content": content})
                for row in rows:
                    if len(row) == 2 and all(row):
                        pages[str(page)].append({"type": "kv", "key": row[0],
                                                 "value": row[1], "content": ""})
            else:
                content = _plain(raw)
                if content:
                    pages[str(page)].append({"type": "text", "y_pos": top,
                                             "x_pos": left, "content": content})
    return pages


def extract_reducto(file_path, api_key):
    """Upload a PDF and parse it with the lower-cost r-1-mini model."""
    if not api_key:
        raise ValueError("REDUCTO_API_KEY is required for EXTRACT_METHOD=reducto")
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(file_path, "rb") as pdf:
        upload = requests.post(
            "https://platform.reducto.ai/upload", headers=headers,
            files={"file": ("document.pdf", pdf, "application/pdf")}, timeout=120)
    if not upload.ok:
        raise RuntimeError(f"Reducto upload failed: HTTP {upload.status_code}")
    file_id = upload.json().get("file_id")
    if not file_id:
        raise ValueError("Reducto upload returned no file ID")
    parsed = requests.post(
        "https://platform.reducto.ai/parse",
        headers={**headers, "Content-Type": "application/json"},
        json={"input": file_id, "settings": {"model": "r-1-mini"}}, timeout=900)
    if not parsed.ok:
        raise RuntimeError(f"Reducto parse failed: HTTP {parsed.status_code}")
    return pages_from_reducto(parsed.json())
