"""
FastAPI backend for GraphRAG chat application.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from neo4j.exceptions import ServiceUnavailable

from api.routers import chat, database, documents, graph, history, classification, chat_tuning, rag_tuning, jobs
from config.settings import settings
from api import auth
from core.singletons import get_graph_db_driver, cleanup_singletons

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# FlashRank prewarm status (timestamps are epoch seconds)
flashrank_prewarm_started_at = None
flashrank_prewarm_completed_at = None
flashrank_prewarm_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application."""
    logger.info("Starting GraphRAG API...")
    # Log resolved LLM provider early to aid debugging misconfigurations
    try:
        logger.info(f"Resolved LLM provider: {settings.llm_provider}")
    except Exception:
        logger.info("Resolved LLM provider: <unavailable>")
    
    # Initialize singletons and connection pool
    try:
        driver = get_graph_db_driver()
        logger.info("Neo4j connection pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j connection pool: {e}")
        # Continue startup for graceful degradation
    
    # Ensure a persistent user token exists and log it once on startup.
    try:
        user_token = auth.ensure_user_token()
        logger.info("Job user token (store this securely if needed): %s", user_token)
    except Exception:
        logger.exception("Failed to ensure user token on startup")
    # Optionally pre-warm the FlashRank reranker if enabled — schedule in background
    try:
        if getattr(settings, "flashrank_enabled", False) and getattr(settings, "flashrank_prewarm_in_process", True):
            logger.info("FlashRank enabled in settings — scheduling background pre-warm of ranker...")
            try:
                # Import locally to avoid bringing optional deps into module import time
                from rag.rerankers.flashrank_reranker import prewarm_ranker
                import threading

                def _background_prewarm():
                    global flashrank_prewarm_started_at, flashrank_prewarm_completed_at, flashrank_prewarm_error
                    try:
                        import time as _time
                        flashrank_prewarm_started_at = _time.time()
                        logger.info("FlashRank background prewarm started")
                        prewarm_ranker()
                        flashrank_prewarm_completed_at = _time.time()
                        logger.info("FlashRank background prewarm completed")
                    except Exception as e:
                        import time as _time
                        flashrank_prewarm_error = str(e)
                        flashrank_prewarm_completed_at = _time.time()
                        logger.warning("FlashRank background prewarm failed: %s", e)

                t = threading.Thread(target=_background_prewarm, name="flashrank-prewarm", daemon=True)
                t.start()
            except Exception as e:
                logger.warning("Failed to schedule FlashRank pre-warm at startup: %s", e)
    except Exception:
        logger.exception("Unexpected error while attempting to schedule FlashRank pre-warm")
    yield
    logger.info("Shutting down GraphRAG API...")
    cleanup_singletons()


app = FastAPI(
    title="GraphRAG API",
    description="Backend API for GraphRAG chat application",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/api/admin/prewarm-status")
async def prewarm_status():
    """Return FlashRank prewarm status for operational visibility."""
    in_progress = bool(flashrank_prewarm_started_at and not flashrank_prewarm_completed_at and not flashrank_prewarm_error)
    return {
        "started_at": flashrank_prewarm_started_at,
        "completed_at": flashrank_prewarm_completed_at,
        "error": flashrank_prewarm_error,
        "in_progress": in_progress,
    }


@app.exception_handler(ServiceUnavailable)
async def neo4j_service_unavailable_handler(request: Request, exc: ServiceUnavailable):
    logger.error("Graph DB unavailable (global handler): %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Graph database unavailable"})

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(database.router, prefix="/api/database", tags=["database"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(graph.router, prefix="/api/graph", tags=["graph"])
app.include_router(history.router, prefix="/api/history", tags=["history"])
app.include_router(classification.router, prefix="/api/classification", tags=["classification"])
app.include_router(chat_tuning.router, prefix="/api/chat-tuning", tags=["chat-tuning"])
app.include_router(rag_tuning.router, prefix="/api/rag-tuning", tags=["rag-tuning"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    flashrank_status = {
        "enabled": bool(getattr(settings, "flashrank_enabled", False)),
        "started_at": flashrank_prewarm_started_at,
        "completed_at": flashrank_prewarm_completed_at,
        "error": flashrank_prewarm_error,
        "in_progress": bool(flashrank_prewarm_started_at and not flashrank_prewarm_completed_at and not flashrank_prewarm_error)
    }
    return {
        "status": "healthy",
        "version": "2.0.0",
        "llm_provider": settings.llm_provider,
        "enable_entity_extraction": settings.enable_entity_extraction,
        "enable_quality_scoring": settings.enable_quality_scoring,
        "flashrank": flashrank_status,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "GraphRAG API",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
