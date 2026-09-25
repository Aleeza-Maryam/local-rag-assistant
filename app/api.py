"""FastAPI endpoints for the RAG assistant."""
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field
import tempfile

from src.ingest import ingest_pdf
from src.retriever import Retriever
from src.generator import generate_answer


app = FastAPI(
    title="Local RAG Assistant API",
    description="Query your PDF documents with citations using RAG.",
    version="1.0.0",
)


# ---------- Schemas ----------
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask")
    provider: str = Field("gemini", description="LLM provider: gemini or groq")
    top_k: int = Field(4, ge=1, le=10, description="Number of chunks to retrieve")


class SourceItem(BaseModel):
    source: str
    page: int
    text: str
    distance: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    provider: str
    sources: List[SourceItem]


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int


class StatsResponse(BaseModel):
    total_chunks: int


# ---------- Global retriever (loaded once) ----------
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# ---------- Routes ----------
@app.get("/")
def root():
    return {
        "name": "Local RAG Assistant API",
        "version": "1.0.0",
        "endpoints": ["/ingest", "/query", "/stats", "/reset", "/docs"],
    }


@app.get("/stats", response_model=StatsResponse)
def stats():
    """Return number of chunks in the vector DB."""
    return StatsResponse(total_chunks=get_retriever().count())


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF and add it to the vector DB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = ingest_pdf(tmp_path)
        for c in chunks:
            c["source"] = file.filename
        added = get_retriever().add_chunks(chunks)
        return IngestResponse(
            filename=file.filename,
            chunks_added=added,
            total_chunks=get_retriever().count(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Ask a question about the ingested documents."""
    retriever = get_retriever()
    if retriever.count() == 0:
        raise HTTPException(status_code=400, detail="No documents in DB. Ingest first.")

    hits = retriever.search(req.query, top_k=req.top_k)

    try:
        answer = generate_answer(req.query, hits, provider=req.provider)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return QueryResponse(
        query=req.query,
        answer=answer,
        provider=req.provider,
        sources=[
            SourceItem(
                source=h["source"],
                page=h["page"],
                text=h["text"][:500],
                distance=h.get("distance"),
            )
            for h in hits
        ],
    )


@app.post("/reset")
def reset():
    """Clear the vector DB."""
    get_retriever().reset()
    return {"status": "reset", "total_chunks": get_retriever().count()}