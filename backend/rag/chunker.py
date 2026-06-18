"""
chunker.py — Text Churning and Chunking
======================================
Second stage of the RAG pipeline. Takes page-by-page text from the loader
and splits it into small, overlapping chunks while keeping track of page metadata.

If the standard langchain module is not fully available in this environment,
we fall back to a custom pure-python implementation of RecursiveCharacterTextSplitter.
"""

from langchain_core.documents import Document

try:
    # Try importing the standard text splitter if available
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    # Custom fallback implementation of RecursiveCharacterTextSplitter
    # designed to split text on logical boundaries with overlap.
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200, length_function=len):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function
            self.separators = ["\n\n", "\n", " ", ""]

        def _split_text(self, text: str, separators: list[str]) -> list[str]:
            if self.length_function(text) <= self.chunk_size:
                return [text]

            if not separators:
                # Character-level split fallback
                chunks = []
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                    chunks.append(text[i:i + self.chunk_size])
                return chunks

            # Find the best separator that appears in this block
            separator = separators[0]
            for s in separators:
                if s == "":
                    separator = s
                    break
                if s in text:
                    separator = s
                    break

            if separator == "":
                splits = list(text)
            else:
                splits = text.split(separator)

            chunks = []
            current_chunk = []
            current_len = 0
            next_separators = separators[1:]

            for split in splits:
                split_len = self.length_function(split)
                
                # If a single split exceeds the chunk size, split it recursively
                if split_len > self.chunk_size:
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                        current_chunk = []
                        current_len = 0
                    
                    sub_splits = self._split_text(split, next_separators)
                    for sub_split in sub_splits:
                        sub_split_len = self.length_function(sub_split)
                        join_len = self.length_function(separator) if current_chunk else 0
                        if current_len + sub_split_len + join_len <= self.chunk_size:
                            current_chunk.append(sub_split)
                            current_len += sub_split_len + join_len
                        else:
                            if current_chunk:
                                chunks.append(separator.join(current_chunk))
                            current_chunk = [sub_split]
                            current_len = sub_split_len
                else:
                    join_len = self.length_function(separator) if current_chunk else 0
                    if current_len + split_len + join_len <= self.chunk_size:
                        current_chunk.append(split)
                        current_len += split_len + join_len
                    else:
                        if current_chunk:
                            chunks.append(separator.join(current_chunk))
                        
                        # Apply overlap by back-tracking on previous splits
                        overlap_splits = []
                        overlap_len = 0
                        for prev_split in reversed(current_chunk):
                            prev_len = self.length_function(prev_split)
                            p_join_len = self.length_function(separator) if overlap_splits else 0
                            if overlap_len + prev_len + p_join_len <= self.chunk_overlap:
                                overlap_splits.insert(0, prev_split)
                                overlap_len += prev_len + p_join_len
                            else:
                                break
                        
                        current_chunk = overlap_splits + [split]
                        current_len = sum(self.length_function(s) for s in current_chunk) + (self.length_function(separator) * (len(current_chunk) - 1))

            if current_chunk:
                chunks.append(separator.join(current_chunk))

            return chunks

        def split_text(self, text: str) -> list[str]:
            return self._split_text(text, self.separators)


def chunk_text(pages: list[str], chunk_size: int = 500, chunk_overlap: int = 100) -> list[Document]:
    """
    Splits page-by-page text into smaller overlapping Document objects.

    Args:
        pages: List of strings, where each string represents the text of one page.
        chunk_size: Maximum size of each text chunk (in characters).
        chunk_overlap: Overlap between consecutive chunks (in characters).

    Returns:
        A list of LangChain Document objects, each containing:
          - page_content: The chunked text string.
          - metadata: A dict containing {"page": page_number} (1-indexed).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks: list[Document] = []

    for i, page_text in enumerate(pages):
        page_num = i + 1

        if not page_text.strip():
            continue

        page_chunks = splitter.split_text(page_text)

        for chunk in page_chunks:
            # We filter out empty chunks to keep the embeddings database clean
            if chunk.strip():
                doc = Document(
                    page_content=chunk.strip(),
                    metadata={
                        "page": page_num
                    }
                )
                chunks.append(doc)

    return chunks
