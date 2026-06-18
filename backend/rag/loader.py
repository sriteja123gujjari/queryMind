"""
loader.py — PDF Text Extraction
================================
First stage of the RAG pipeline. Takes raw PDF bytes (from a FastAPI upload)
and extracts the text content from every page.

Why bytes instead of a file path?
  - FastAPI gives us uploaded files as in-memory bytes (via UploadFile.read()).
  - Accepting bytes means we never have to save the PDF to disk, which is
    cleaner, faster, and avoids temp-file cleanup headaches.
"""

import fitz  # PyMuPDF — imported as "fitz" for historical reasons (the original C library was called MuPDF/Fitz)


def load_pdf(file_bytes: bytes) -> list[str]:
    """
    Extract text from a PDF, returning one string per page.

    Args:
        file_bytes: The raw bytes of the PDF file.

    Returns:
        A list of strings, where each string is the text content of one page.
        Empty pages are included (as empty strings) to preserve page numbering.

    Raises:
        ValueError: If the PDF has zero pages or contains no extractable text.
    """

    # fitz.open() can accept a file path OR raw bytes.
    # When passing bytes, we MUST specify filetype="pdf" — otherwise fitz
    # tries to guess from the (non-existent) file extension and may fail.
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    if doc.page_count == 0:
        doc.close()
        raise ValueError("PDF has zero pages — is the file corrupted?")

    pages: list[str] = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # 0-indexed internally

        # get_text("text") returns plain text with layout preserved.
        # Other options: "html", "dict" (structured), "blocks" (paragraphs).
        # Plain text is what we want — the chunker doesn't need HTML tags.
        text = page.get_text("text")

        # Strip leading/trailing whitespace per page to keep things clean.
        pages.append(text.strip())

    doc.close()

    # Safety check: if EVERY page is empty, the PDF is probably scanned images.
    # Better to fail loudly now than silently return nothing.
    if not any(pages):
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned document (images of text). "
            "OCR support is not yet implemented."
        )

    return pages
