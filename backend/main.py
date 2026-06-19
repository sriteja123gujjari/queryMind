"""
main.py — FastAPI Backend and RAG Orchestrator
==============================================
Central hub of QueryMind. Provides endpoints for PDF/Code/ZIP uploads,
context retrieval from ChromaDB, and querying LLMs (Groq, Ollama, Gemini, Claude).
"""

import os
import io
import zipfile
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

# Import our RAG stages
from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.code_chunker import chunk_code_file
from rag.retriever import add_documents, retrieve_context, clear_store, get_chunk_count

# Load environment variables
load_dotenv()

app = FastAPI(title="QueryMind Backend API", version="1.0")

# Enable CORS for the React frontend
# ALLOWED_ORIGINS env var can be a comma-separated list of origins, e.g.:
#   "https://querymind.vercel.app,http://localhost:5173"
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = (
    ["*"] if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document status metadata cache
document_metadata = {
    "filename": None,
    "file_type": None,  # "pdf", "code", or "zip"
    "page_count": 0,    # represents page count for PDF, or file count for code/zip
    "chunk_count": 0,
}

# --- Request Models ---
class QueryRequest(BaseModel):
    query: str
    provider: str        # "groq", "ollama", "gemini", or "claude"
    apiKey: Optional[str] = None
    baseUrl: Optional[str] = None  # for Ollama, e.g. "http://localhost:11434"
    model: Optional[str] = None    # e.g., "llama-3.3-70b-versatile" or "gemini-1.5-flash"


# --- LLM Client Helpers ---
def query_groq(prompt: str, system_prompt: str, api_key: str, model: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model or "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"Groq API Error: {response.text}"
        )
    return response.json()["choices"][0]["message"]["content"]


def query_ollama(prompt: str, system_prompt: str, base_url: str, model: str) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model or "llama3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "options": {
            "temperature": 0.0
        },
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=45)
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama connection failed at {url}. Is Ollama running? Error: {str(e)}"
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Ollama API Error: {response.text}"
        )
    return response.json()["message"]["content"]


def query_gemini(prompt: str, system_prompt: str, api_key: str, model: str) -> str:
    model_name = model or "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    # We combine system prompt and user query into contents for maximum compatibility
    full_prompt = f"{system_prompt}\n\nRetrieved context and question details:\n{prompt}"
    
    payload = {
        "contents": [
            {
                "parts": [{"text": full_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0
        }
    }
    response = requests.post(url, json=payload, timeout=30)
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Gemini API Error: {response.text}"
        )
    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse Gemini API response. Response details: {response.text}"
        )


def query_claude(prompt: str, system_prompt: str, api_key: str, model: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": model or "claude-3-5-sonnet-latest",
        "max_tokens": 1500,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Claude API Error: {response.text}"
        )
    return response.json()["content"][0]["text"]


# --- Endpoints ---

@app.get("/api/status")
async def get_status():
    """
    Get current workspace metadata status (file type, name, chunks).
    Also checks active ChromaDB count.
    """
    chunk_count = get_chunk_count()
    if chunk_count == 0:
        # If DB is empty, clear in-memory state
        document_metadata["filename"] = None
        document_metadata["file_type"] = None
        document_metadata["page_count"] = 0
        document_metadata["chunk_count"] = 0
    else:
        document_metadata["chunk_count"] = chunk_count
        
    return document_metadata


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, Code, or ZIP archive).
    Parses, chunks, and indexes it into ChromaDB.
    """
    filename = file.filename
    file_bytes = await file.read()
    
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
        
    documents = []
    file_type = None
    page_count = 0
    
    # 1. Process ZIP File (CodeMode)
    if filename.lower().endswith(".zip"):
        zip_buffer = io.BytesIO(file_bytes)
        try:
            with zipfile.ZipFile(zip_buffer) as z:
                for zip_info in z.infolist():
                    if zip_info.is_dir():
                        continue
                    
                    # Ignore common configuration/dependency files
                    parts = zip_info.filename.split('/')
                    if any(p.startswith('.') or p in ('__pycache__', 'node_modules', 'dist', 'build', 'venv', 'env') for p in parts):
                        continue
                    
                    # Only parse supported extensions
                    _, ext = os.path.splitext(zip_info.filename.lower())
                    supported_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.go', '.cpp', '.cc', '.h', '.java', '.html', '.css', '.rb', '.rs', '.txt', '.md', '.json', '.yaml', '.yml']
                    if ext not in supported_extensions:
                        continue
                        
                    try:
                        with z.open(zip_info) as f:
                            content = f.read().decode('utf-8', errors='ignore')
                            if not content.strip() or len(content) > 500000: # ignore empty or huge files
                                continue
                            file_docs = chunk_code_file(content, zip_info.filename)
                            if file_docs:
                                documents.extend(file_docs)
                                page_count += 1  # count files
                    except Exception:
                        continue
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse ZIP file: {str(e)}")
            
        if not documents:
            raise HTTPException(status_code=400, detail="No readable source code files found in the ZIP archive.")
        file_type = "zip"
        
    # 2. Process Individual Code File (CodeMode)
    elif any(filename.lower().endswith(ext) for ext in ['.py', '.js', '.jsx', '.ts', '.tsx', '.go', '.cpp', '.cc', '.h', '.java', '.html', '.css', '.rb', '.rs', '.txt', '.md', '.json', '.yaml', '.yml']):
        try:
            content = file_bytes.decode('utf-8', errors='ignore')
            documents = chunk_code_file(content, filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract contents from code file.")
        file_type = "code"
        page_count = 1

    # 3. Process PDF File (DocMode)
    elif filename.lower().endswith(".pdf"):
        try:
            pages = load_pdf(file_bytes)
            documents = chunk_text(pages)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract text chunks from PDF.")
        file_type = "pdf"
        page_count = len(pages)
        
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload a PDF, a supported code file, or a ZIP codebase archive."
        )
        
    # Reset ChromaDB and add the new documents
    clear_store()
    add_documents(documents)
    
    # Update state
    document_metadata["filename"] = filename
    document_metadata["file_type"] = file_type
    document_metadata["page_count"] = page_count
    document_metadata["chunk_count"] = len(documents)
    
    return document_metadata


@app.post("/api/query")
async def query_pipeline(request: QueryRequest):
    """
    Search indexed context and answer user query using the specified LLM.
    Ensures that the answer is sourced strictly from the documents.
    """
    # Verify vector store is populated
    chunk_count = get_chunk_count()
    if chunk_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No document has been uploaded. Please upload a file first."
        )
        
    # Retrieve relevant context
    retrieved_docs = retrieve_context(request.query, k=5)
    if not retrieved_docs:
        raise HTTPException(
            status_code=404,
            detail="No relevant context could be found in the database."
        )
        
    # Build context string and source references list
    context_str = ""
    sources = []
    
    for idx, doc in enumerate(retrieved_docs):
        meta = doc.metadata
        source_label = ""
        
        if "page" in meta:
            source_label = f"Page {meta['page']}"
        elif "source" in meta:
            start_line = meta.get("start_line", 1)
            end_line = meta.get("end_line", 1)
            source_label = f"{meta['source']} (Lines {start_line}-{end_line})"
            
        context_str += f"--- Context Block {idx+1} [{source_label}] ---\n{doc.page_content}\n\n"
        
        sources.append({
            "id": idx + 1,
            "content": doc.page_content,
            "source": meta.get("source", document_metadata["filename"]),
            "page": meta.get("page"),
            "start_line": meta.get("start_line"),
            "end_line": meta.get("end_line"),
            "language": meta.get("language", "text")
        })
        
    # Build System Prompt and User Prompts
    system_prompt = (
        "You are a helpful, precise RAG assistant. You must answer the user's question "
        "based ONLY on the provided context retrieved from the document/codebase.\n"
        "Ensure you follow these rules strictly:\n"
        "1. Answer the question using ONLY the retrieved context. Do NOT make up information or use external knowledge.\n"
        "2. If the answer cannot be found in the context, respond EXACTLY with: "
        "'I cannot find the answer in the uploaded context.' and do NOT add any further explanation.\n"
        "3. Keep the explanations concise, clear, and professional. Mention the sources/files where relevant.\n"
        "4. For programming queries, provide complete functional code snippets derived from the context."
    )
    
    user_prompt = (
        f"Retrieved Context:\n"
        f"=========================================\n"
        f"{context_str}"
        f"=========================================\n\n"
        f"User Question: {request.query}\n\n"
        f"Answer:"
    )
    
    # Resolve API Key / Base URL defaults from env if not provided in request
    provider = request.provider.lower()
    api_key = request.apiKey
    base_url = request.baseUrl
    model = request.model
    
    if provider == "groq":
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise HTTPException(status_code=400, detail="Groq API Key is missing. Please provide it in settings.")
        answer = query_groq(user_prompt, system_prompt, key, model)
        
    elif provider == "ollama":
        url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        answer = query_ollama(user_prompt, system_prompt, url, model)
        
    elif provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise HTTPException(status_code=400, detail="Gemini API Key is missing. Please provide it in settings.")
        answer = query_gemini(user_prompt, system_prompt, key, model)
        
    elif provider == "claude":
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise HTTPException(status_code=400, detail="Claude API Key is missing. Please provide it in settings.")
        answer = query_claude(user_prompt, system_prompt, key, model)
        
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider}")
        
    return {
        "answer": answer.strip(),
        "sources": sources
    }


@app.post("/api/clear")
async def clear_pipeline():
    """
    Clear all uploaded document state and purge the Chroma database.
    """
    success = clear_store()
    document_metadata["filename"] = None
    document_metadata["file_type"] = None
    document_metadata["page_count"] = 0
    document_metadata["chunk_count"] = 0
    
    if not success:
         raise HTTPException(status_code=500, detail="Failed to completely remove the vector database files.")
         
    return {"status": "cleared"}
