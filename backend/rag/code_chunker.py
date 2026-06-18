"""
code_chunker.py — Code-Aware Structured Text Splitting
======================================================
Second stage of the RAG pipeline for CodeMode. Handles splitting source code
files into logical chunks matching function and class boundaries instead of
arbitrary character limits.
"""

import os
from langchain_core.documents import Document

try:
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
except ImportError:
    # Fallback to langchain.text_splitter if langchain_text_splitters isn't installed
    from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

# Mapping of file extensions to LangChain Language enums
EXTENSION_MAPPING = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".go": Language.GO,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".h": Language.CPP,
    ".java": Language.JAVA,
    ".html": Language.HTML,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
}

def chunk_code_file(file_content: str, filename: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    """
    Split a single source code file into logical, language-aware chunks.
    Tracks file name, programming language, and estimated start/end lines.

    Args:
        file_content: The full text content of the source file.
        filename: The filename (including path) of the file.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        A list of Document objects with structured metadata.
    """
    _, ext = os.path.splitext(filename.lower())
    lang = EXTENSION_MAPPING.get(ext, None)

    if lang:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    chunks = splitter.split_text(file_content)
    documents = []

    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        # Find start and end line numbers of this chunk in the original source
        start_line = 1
        end_line = 1
        try:
            char_idx = file_content.find(chunk)
            if char_idx != -1:
                start_line = file_content[:char_idx].count('\n') + 1
                end_line = start_line + chunk.count('\n')
        except Exception:
            pass

        doc = Document(
            page_content=chunk.strip(),
            metadata={
                "source": filename,
                "language": lang.value if lang else "text",
                "start_line": start_line,
                "end_line": end_line,
                "chunk_index": idx
            }
        )
        documents.append(doc)

    return documents
