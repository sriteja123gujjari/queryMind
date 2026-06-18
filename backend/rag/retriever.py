"""
retriever.py — Vector Store and Context Retrieval
==================================================
Fourth stage of the RAG pipeline. Manages the ChromaDB vector database,
indexing text/code chunks and retrieving relevant context for queries.
"""

import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from rag.embedder import get_embedder

# Setup database persistence directory in the backend folder
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.getenv("CHROMA_DB_PATH", os.path.join(BACKEND_DIR, "chroma_db"))
COLLECTION_NAME = "querymind_collection"

_db_instance = None


def _make_store() -> Chroma:
    """Create a fresh Chroma vector store connection."""
    embedder = get_embedder()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=DB_DIR,
    )


def get_vector_store() -> Chroma:
    """
    Get or initialize the persistent Chroma vector store instance.
    Reconnects automatically if the instance is stale (e.g. after server reload).
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = _make_store()
    return _db_instance


def add_documents(documents: list[Document]) -> bool:
    """
    Index a list of Document objects in the vector store.
    """
    if not documents:
        return False
    db = get_vector_store()
    db.add_documents(documents)
    return True


def retrieve_context(query: str, k: int = 5) -> list[Document]:
    """
    Search for documents semantically similar to the query.

    Args:
        query: The user's prompt or question.
        k: The number of relevant documents to retrieve.

    Returns:
        A list of LangChain Document objects.
    """
    db = get_vector_store()
    return db.similarity_search(query, k=k)


def clear_store() -> bool:
    """
    Reset and clear the Chroma database completely.
    Deletes the collection in-place (avoids Windows file-locking issues)
    and resets the in-memory reference so the next call creates a fresh store.
    """
    global _db_instance

    try:
        # Use the existing instance to delete the collection if possible
        if _db_instance is not None:
            try:
                _db_instance.delete_collection()
            except Exception:
                pass
            _db_instance = None

        # Recreate a fresh empty store (creates a new collection)
        _db_instance = _make_store()
        return True

    except Exception as e:
        print(f"Warning: clear_store encountered an error: {e}")
        _db_instance = None
        return False


def get_chunk_count() -> int:
    """
    Get the total number of document chunks currently indexed.
    Returns 0 on any error (e.g. stale collection after reload).
    """
    try:
        db = get_vector_store()
        if db and db._collection:
            return db._collection.count()
    except Exception:
        # Collection may not exist yet — that's fine
        pass
    return 0
