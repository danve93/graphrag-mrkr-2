"""
Database management utilities for TruLens.

Provides functions for:
- Database connection management
- Backup and restore
- Cleanup and maintenance
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_database_url(config: Optional[dict] = None) -> str:
    """
    Get TruLens database URL from configuration.

    Args:
        config: Configuration dict (optional)

    Returns:
        str: SQLAlchemy database URL
    """
    if config is None:
        # Default to SQLite
        return "sqlite:///evals/trulens/trulens.db"

    db_config = config.get("trulens", {}).get("database", {})
    backend = db_config.get("backend", "sqlite")

    if backend == "postgresql":
        pg = db_config.get("postgresql", {})
        host = pg.get("host", "localhost")
        port = pg.get("port", 5432)
        database = pg.get("database", "trulens")
        user = pg.get("user", "postgres")
        password = pg.get("password", "")
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    else:
        sqlite = db_config.get("sqlite", {})
        path = sqlite.get("path", "evals/trulens/trulens.db")
        return f"sqlite:///{path}"


def reset_database(database_url: str):
    """
    Reset TruLens database (clear all records).

    Args:
        database_url: SQLAlchemy database URL
    """
    try:
        from trulens_eval import Tru

        tru = Tru(database_url=database_url)
        tru.reset_database()

        logger.info(f"Database reset: {database_url}")

    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise


def backup_database(database_url: str, backup_path: str):
    """
    Backup TruLens database.

    Args:
        database_url: Source database URL
        backup_path: Backup file path
    """
    if database_url.startswith("sqlite:///"):
        # SQLite: simple file copy
        import shutil
        source_path = database_url.replace("sqlite:///", "")
        shutil.copy2(source_path, backup_path)
        logger.info(f"Database backed up: {source_path} -> {backup_path}")
    else:
        # PostgreSQL: use pg_dump (requires external tool)
        logger.warning("PostgreSQL backup requires pg_dump utility")
        raise NotImplementedError("PostgreSQL backup not implemented")
