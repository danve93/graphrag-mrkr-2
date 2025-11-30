#!/usr/bin/env python3
"""
Simple worker script to run FlashRank prewarm out-of-process.

Usage (inside backend container):
    python scripts/flashrank_prewarm_worker.py

This script will import the reranker module and call `prewarm_ranker()`.
It reads configuration from `config.settings` and will no-op if `flashrank_enabled` is false.
"""
import logging
import sys
import os
from pathlib import Path

# Remove environment variables that can interfere with pydantic Settings
# (some deployment `.env` files set `NEO4J_AUTH` for the Neo4j container,
# which is not a field on our Settings model and causes strict validation to fail).
os.environ.pop("NEO4J_AUTH", None)
# Some deployments place a `.env` file at the project root with keys intended
# for other containers (e.g., `NEO4J_AUTH=neo4j/password` for the Neo4j image).
# The Settings model is configured to load `.env` and is strict about unknown
# variables; to avoid validation failures, temporarily sanitize the `.env`
# file while importing settings in this worker process.
env_path = Path("/app/.env")
backup_path = Path("/tmp/.env.worker.bak")
try:
    if env_path.exists():
        env_path.replace(backup_path)
        # create a clean, empty .env so pydantic doesn't pick up unrelated keys
        env_path.write_text("")
except Exception:
    # If backup or write fails, proceed — we still attempt import and will
    # catch validation errors below.
    pass

from config.settings import settings

# After importing settings, restore the original .env if we backed it up.
try:
    if backup_path.exists():
        backup_path.replace(env_path)
except Exception:
    pass

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger("flashrank-worker")


def main():
    if not getattr(settings, "flashrank_enabled", False):
        logger.info("FlashRank is disabled in settings; exiting.")
        return 0

    logger.info("Starting FlashRank prewarm worker...")
    try:
        # Import locally so optional deps are only required here
        from rag.rerankers.flashrank_reranker import prewarm_ranker

        prewarm_ranker()
        logger.info("FlashRank prewarm completed successfully")
        return 0
    except Exception as exc:
        logger.exception("FlashRank prewarm worker failed: %s", exc)
        return 2


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
