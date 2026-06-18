"""
Integration test for embedder.py, code_chunker.py, and retriever.py
==================================================================
Run from backend/ folder:
    python test_rag.py
"""

from langchain_core.documents import Document
from rag.embedder import get_embedder
from rag.code_chunker import chunk_code_file
from rag.retriever import get_vector_store, add_documents, retrieve_context, clear_store, get_chunk_count

def test_rag_pipeline():
    print("=" * 60)
    print("TEST: RAG Pipeline Integration")
    print("=" * 60)

    # 1. Reset database
    print("Resetting database...")
    clear_store()
    assert get_chunk_count() == 0, "Database should be empty after clearing"
    print("  [PASS] Database cleared successfully.")

    # 2. Test Embedder Initialization
    print("Initializing embedder...")
    embedder = get_embedder()
    assert embedder is not None, "Embedder failed to load"
    print("  [PASS] Embedder loaded successfully.")

    # 3. Test Code Churning and Chunking
    print("Chunking sample python file...")
    dummy_code = (
        "class RentalSystem:\n"
        "    def __init__(self):\n"
        "        self.auth = 'supabase'\n"
        "\n"
        "    def get_auth_token(self):\n"
        "        return 'token_12345'\n"
        "\n"
        "    def verify_session(self):\n"
        "        return True\n"
    )
    code_docs = chunk_code_file(dummy_code, "rental_system.py")
    assert len(code_docs) > 0, "No chunks generated from code"
    assert code_docs[0].metadata["source"] == "rental_system.py", "Metadata source mismatch"
    assert code_docs[0].metadata["language"] == "python", "Language identification mismatch"
    print(f"  [PASS] Code chunked successfully into {len(code_docs)} chunks.")

    # 4. Add Documents to Vector Store
    print("Indexing chunks into ChromaDB...")
    success = add_documents(code_docs)
    assert success is True, "Failed to index documents"
    chunk_count = get_chunk_count()
    assert chunk_count == len(code_docs), f"Chunk count mismatch: expected {len(code_docs)}, got {chunk_count}"
    print(f"  [PASS] Chunks indexed successfully. Total chunks in store: {chunk_count}")

    # 5. Retrieve Context
    print("Retrieving relevant context for query 'supabase auth token'...")
    results = retrieve_context("supabase auth token", k=1)
    assert len(results) > 0, "No results retrieved"
    assert "supabase" in results[0].page_content.lower(), "Retrieved content does not match query"
    print(f"  [PASS] Retrieve successful. Closest match source: {results[0].metadata['source']}")

    # 6. Clear Store
    print("Clearing store...")
    clear_store()
    assert get_chunk_count() == 0, "Store should be empty after clearing"
    print("  [PASS] Store cleared successfully.")
    
    print("=" * 60)
    print("RAG PIPELINE INTEGRATION PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_rag_pipeline()
