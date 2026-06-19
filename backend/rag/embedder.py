"""
embedder.py — Local Offline Embeddings (Lightweight ONNX version)
===================================================================
Third stage of the RAG pipeline. Provides text embeddings using ChromaDB's
bundled ONNX MiniLM model instead of the full PyTorch/sentence-transformers
stack.

Why the switch from sentence-transformers?
  - sentence-transformers pulls in PyTorch, which alone uses 300-500MB+ of
    RAM just to import — this blows past free-tier hosting limits (e.g.
    Render's 512MB cap) and causes OOM crashes on startup.
  - ChromaDB ships its own ONNX-runtime-based MiniLM-L6-v2 embedding
    function as a built-in default. It produces equivalent 384-dim
    embeddings using onnxruntime, which has a dramatically smaller memory
    footprint than torch.

This file wraps chromadb's embedding function in a small adapter class so
it's compatible with langchain's `Embeddings` interface (embed_documents /
embed_query), which `langchain_community.vectorstores.Chroma` expects.
"""

from typing import List
from langchain_core.embeddings import Embeddings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

# Cache the embedder instance to avoid reloading it on every call
_embedder_instance = None


class ChromaONNXEmbeddings(Embeddings):
    """
    Adapter that exposes ChromaDB's lightweight ONNX MiniLM embedding
    function through langchain's Embeddings interface.
    """

    def __init__(self):
        # This downloads/loads the ONNX model on first instantiation
        # (~80MB on disk), much lighter at runtime than torch.
        self._ef = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._ef(texts)
        # chromadb's embedding functions can return numpy arrays; normalize to plain lists
        return [list(map(float, vec)) for vec in embeddings]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embedder() -> ChromaONNXEmbeddings:
    """
    Returns a cached instance of the lightweight ONNX embedder.
    Loads the bundled MiniLM-L6-v2 ONNX model locally on first call.
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = ChromaONNXEmbeddings()
    return _embedder_instance
