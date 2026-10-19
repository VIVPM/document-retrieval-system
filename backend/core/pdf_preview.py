import fitz


def page_png(data: bytes, page: int) -> tuple[bytes, int]:
    with fitz.open(stream=data, filetype="pdf") as pdf:
        if page < 1 or page > pdf.page_count:
            raise IndexError("Page not found.")
        current = pdf[page - 1]
        scale = min(2, 1600 / current.rect.width, 2000 / current.rect.height)
        return current.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png"), pdf.page_count


if __name__ == "__main__":
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    data = doc.tobytes()
    image, count = page_png(data, 2)
    assert image.startswith(b"\x89PNG\r\n\x1a\n") and count == 2
    for invalid in (0, 3):
        try:
            page_png(data, invalid)
        except IndexError:
            continue
        raise AssertionError(f"page {invalid} should not render")
    print("pdf preview selftest OK")
