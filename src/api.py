"""FastAPI interface for the RAG chatbot.

Run with:  uvicorn src.api:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.logger import get_logger
from src.vectorstore import get_pinecone_client, VectorStore, VectorStoreError
from src.embeddings import Embedder, EmbeddingError
from src.llm import LLMClient, LLMError
from src.graph import build_graph, run_query

log = get_logger(__name__)

app = FastAPI(
    title="Agentic AI Ebook RAG Chatbot",
    description="Answers questions strictly from Konverge.ai's Agentic AI ebook.",
    version="1.0.0",
)

_pipeline = None
_init_error = None


def _init_pipeline():
    global _pipeline, _init_error
    try:
        pc = get_pinecone_client()
        vectorstore = VectorStore(pc)
        vectorstore.ensure_index()
        embedder = Embedder(pc)
        llm_client = LLMClient()
        _pipeline = build_graph(embedder, vectorstore, llm_client)
        _init_error = None
        log.info("Pipeline initialized.")
    except (VectorStoreError, LLMError, EmbeddingError) as e:
        _init_error = str(e)
        log.error(f"Pipeline init failed: {e}")


@app.on_event("startup")
def startup():
    _init_pipeline()


class ChatRequest(BaseModel):
    question: str = Field(..., description="Question about the Agentic AI ebook")


class Source(BaseModel):
    chunk_id: str
    page_number: int | None
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float
    groundedness: float | None
    used_fallback: bool


@app.get("/health")
def health():
    return {
        "status": "ok" if _pipeline is not None else "degraded",
        "pipeline_ready": _pipeline is not None,
        "init_error": _init_error,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if _pipeline is None:
        _init_pipeline()
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {_init_error or 'unknown initialization error'}",
        )

    try:
        result = run_query(_pipeline, req.question)
    except Exception as e:
        log.error(f"Unhandled error while answering query: {e}")
        raise HTTPException(status_code=500, detail="Internal error while generating the answer.")

    return JSONResponse(content={
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
        "groundedness": result["groundedness"],
        "used_fallback": result["used_fallback"],
    })
