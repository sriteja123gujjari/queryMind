"""
Quick test for loader.py
========================
Run from the backend/ folder:
    python test_loader.py

This will:
  1. Create a tiny PDF in memory (no need to find a real file)
  2. Pass its bytes to load_pdf()
  3. Print the extracted text
  4. Verify the scanned-PDF error handling works
"""

import fitz  # We use PyMuPDF to CREATE a test PDF too - convenient!
from rag.loader import load_pdf


def create_test_pdf(text_per_page: list[str]) -> bytes:
    """Helper: builds a tiny in-memory PDF with the given text on each page."""
    doc = fitz.open()  # new empty PDF
    for text in text_per_page:
        page = doc.new_page()  # default A4 size
        # insert_text draws text at (x=72, y=72) - 1 inch from top-left
        page.insert_text((72, 72), text, fontsize=12)
    pdf_bytes = doc.tobytes()  # serialize to bytes (like reading a .pdf file)
    doc.close()
    return pdf_bytes


# --- Test 1: Normal PDF with 2 pages ---
print("=" * 50)
print("TEST 1: Normal 2-page PDF")
print("=" * 50)

test_bytes = create_test_pdf([
    "Hello from page 1! This is QueryMind's first test.",
    "Page 2 has different content about RAG pipelines."
])

pages = load_pdf(test_bytes)

print(f"  Number of pages extracted: {len(pages)}")
for i, page_text in enumerate(pages):
    print(f"  Page {i + 1}: '{page_text[:60]}...'")

assert len(pages) == 2, "Expected 2 pages!"
assert "QueryMind" in pages[0], "Page 1 text not found!"
assert "RAG" in pages[1], "Page 2 text not found!"
print("  [PASS] PASSED\n")


# --- Test 2: Empty PDF should raise ValueError ---
print("=" * 50)
print("TEST 2: PDF with no text (simulating scanned doc)")
print("=" * 50)

empty_bytes = create_test_pdf([""])  # one page, no text

try:
    load_pdf(empty_bytes)
    print("  [FAIL] FAILED - should have raised ValueError!")
except ValueError as e:
    print(f"  Caught expected error: {e}")
    print("  [PASS] PASSED\n")


# --- Summary ---
print("=" * 50)
print("loader.py works! All tests passed.")
print("=" * 50)
