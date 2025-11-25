"""
FastAPI backend for GraphRAG chat application.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, database, documents, graph, history, classification, chat_tuning, jobs
from config.settings import settings
from api import auth

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application."""
    logger.info("Starting GraphRAG API...")
    # Log resolved LLM provider early to aid debugging misconfigurations
    try:
        logger.info(f"Resolved LLM provider: {settings.llm_provider}")
    except Exception:
        logger.info("Resolved LLM provider: <unavailable>")
    # Ensure a persistent user token exists and log it once on startup.
    try:
        user_token = auth.ensure_user_token()
        logger.info("Job user token (store this securely if needed): %s", user_token)
    except Exception:
        logger.exception("Failed to ensure user token on startup")
    yield
    logger.info("Shutting down GraphRAG API...")


app = FastAPI(
    title="GraphRAG API",
    description="Backend API for GraphRAG chat application",
    version="2.0.0",
    lifespan=lifespan,
)

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
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "llm_provider": settings.llm_provider,
        "enable_entity_extraction": settings.enable_entity_extraction,
        "enable_quality_scoring": settings.enable_quality_scoring,
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
