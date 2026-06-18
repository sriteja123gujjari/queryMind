"""
embedder.py — Local Offline Embeddings
=====================================
Third stage of the RAG pipeline. Provides text embeddings using a local
HuggingFace sentence-transformer model (all-MiniLM-L6-v2).

No external API keys are required for embedding, making it 100% free and offline.
"""

from langchain_huggingface import HuggingFaceEmbeddings

# Cache the embedder instance to avoid reloading it on every call
_embedder_instance = None

def get_embedder() -> HuggingFaceEmbeddings:
    """
    Returns a cached instance of HuggingFaceEmbeddings.
    Loads the 'all-MiniLM-L6-v2' model locally on first call.
    """
    global _embedder_instance
    if _embedder_instance is None:
        # We use all-MiniLM-L6-v2: it is fast, lightweight (384-dimensional),
        # and runs beautifully on standard CPU environments.
        _embedder_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embedder_instance
