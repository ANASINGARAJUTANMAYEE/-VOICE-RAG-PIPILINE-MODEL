"""
FastAPI Server for Voice-Enabled Guardrailed RAG Pipeline
Provides endpoints for audio/text queries, latency benchmarking, and system health status.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.harness.orchestrator import RAGOrchestrator
from src.harness.models import PipelineResponse
from src.retrieval.embed import get_embedder

# Module-level orchestrator (populated in lifespan)
orchestrator: RAGOrchestrator = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the embedding model and initialize orchestrator at startup."""
    global orchestrator
    import time
    print("[startup] Initializing RAG Orchestrator...", flush=True)
    orchestrator = RAGOrchestrator()

    # Force-load the embedding model now so the first user request
    # doesn't pay the ~20s model-load cold-start cost.
    if orchestrator.hybrid_retriever:
        t0 = time.perf_counter()
        print("[startup] Pre-warming embedding model...", flush=True)
        _ = orchestrator.hybrid_retriever.embedder.model  # triggers SentenceTransformer load
        # Run one dummy embed to JIT-compile any torch ops
        orchestrator.hybrid_retriever.embedder.embed_query("warmup")
        warmup_ms = round((time.perf_counter() - t0) * 1000)
        print(f"[startup] Embedding model warm — first embed took {warmup_ms}ms. Ready.", flush=True)
    else:
        print("[startup] No retrieval index found — skipping embedder pre-warm.", flush=True)

    yield  # server is running
    print("[shutdown] RAG server shutting down.", flush=True)


app = FastAPI(
    title="Voice-Enabled Multilingual RAG API",
    description="Guardrailed multilingual RAG (Hindi/Bengali/Tamil) with Sarvam STT (saarika:v2.5) and Google Gemini (gemini-3.7-flash)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local and web testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextQueryRequest(BaseModel):
    query: str
    language: str = "hi"
    top_k: int = 5


@app.get("/")
def serve_index():
    index_path = PROJECT_ROOT / "app" / "static" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Voice-Enabled RAG API is live. Access static UI at /static/index.html"}


@app.get("/api/health")
def health_check():
    import time
    vector_count = 0
    bm25_count = 0
    embed_ms = None

    if orchestrator and orchestrator.hybrid_retriever:
        vector_count = orchestrator.hybrid_retriever.vector_store.total_vectors
        bm25_count = orchestrator.hybrid_retriever.bm25_store.total_docs
        # Live embed latency probe (model already warm)
        t0 = time.perf_counter()
        orchestrator.hybrid_retriever.embedder.embed_query("health check")
        embed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "status": "healthy",
        "languages_supported": ["hi", "bn", "ta"],
        "retrieval_engine": {
            "faiss_vectors": vector_count,
            "bm25_docs": bm25_count,
            "index_type": "FlatIP (brute-force cosine)",
            "embed_latency_ms": embed_ms,
            # Honest note: embedding on CPU dominates retrieval latency (~500ms).
            # P100 cold-start outlier eliminated by pre-warm at startup.
            "retrieval_note": "CPU-bound transformer inference ~500ms warm; GPU would achieve <50ms"
        },
        "speech_engine": "Sarvam AI (saarika:v2.5)",
        "generation_engine": "Google Gemini (gemini-3.1-flash-lite)",
        "guardrails": ["input_safety", "retrieval_confidence", "grounding_anti_hallucination"]
    }


@app.post("/api/query", response_model=PipelineResponse)
async def query_pipeline(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    language: str = Form("hi"),
    top_k: int = Form(5)
):
    """
    Main endpoint accepting audio file or text query.
    Executes full pipeline: STT -> Input Guard -> Hybrid Retrieval -> Ret Guard -> LLM -> Ground Guard.
    """
    try:
        audio_bytes = None
        filename = "audio.webm"
        if file is not None:
            audio_bytes = await file.read()
            filename = file.filename or "audio.webm"

        if not audio_bytes and not text:
            raise HTTPException(status_code=400, detail="Either 'file' (audio) or 'text' must be provided.")

        response = orchestrator.process_query(
            audio_bytes=audio_bytes,
            text_query=text,
            language_hint=language,
            audio_filename=filename,
            top_k=top_k
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query-text", response_model=PipelineResponse)
def query_text_only(request: TextQueryRequest):
    """
    Direct text query endpoint for latency testing and evaluation without audio encoding overhead.
    """
    try:
        response = orchestrator.process_query(
            text_query=request.query,
            language_hint=request.language,
            top_k=request.top_k
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static directory
static_dir = PROJECT_ROOT / "app" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)

