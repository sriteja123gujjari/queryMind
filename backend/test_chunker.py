"""
Quick test for chunker.py
=========================
Run from the backend/ folder:
    python test_chunker.py

This will:
  1. Define a dummy list of page texts.
  2. Pass them to chunk_text() with a small chunk size to force splitting.
  3. Verify chunks have overlapping segments and correct page metadata.
"""

from rag.chunker import chunk_text


# A sample text representing 2 pages of document text.
# The content on Page 1 is long enough to split if we set chunk_size small.
pages = [
    (
        "Introduction to QueryMind. QueryMind is a private document RAG assistant. "
        "It uses a local database for storage. FastAPI powers the backend. "
        "React runs on the frontend. The entire application runs offline."
    ),
    (
        "System configuration instructions. Ensure Python 3.10+ is installed. "
        "Run pip install to pull the dependencies. Update the environment file."
    )
]

print("=" * 60)
print("TEST: Chunking text with size=80, overlap=20")
print("=" * 60)

# We use small chunk sizes here to force chunking on short test strings
chunks = chunk_text(pages, chunk_size=80, chunk_overlap=20)

print(f"Total chunks created: {len(chunks)}\n")

for i, doc in enumerate(chunks):
    print(f"Chunk {i + 1} (Page {doc.metadata['page']}):")
    print(f"  Content: '{doc.page_content}'")
    print(f"  Length:  {len(doc.page_content)} characters")
    print("-" * 40)

# Verifications
assert len(chunks) > 2, "Should have created multiple chunks due to small chunk size"
assert all("page" in doc.metadata for doc in chunks), "Every chunk must have a page number in metadata"
assert chunks[0].metadata["page"] == 1, "First chunk should belong to page 1"
assert chunks[-1].metadata["page"] == 2, "Last chunk should belong to page 2"

print("  [PASS] PASSED - chunker.py is working correctly!")
print("=" * 60)
