"""
TruLens initialization module (opt-in).

Singleton initialization for TruLens continuous monitoring.
Call initialize_trulens() once at application startup if monitoring is enabled.

Usage:
    # In api/main.py startup event
    if os.getenv("ENABLE_TRULENS_MONITORING") == "1":
        from evals.trulens.trulens_initializer import initialize_trulens
        initialize_trulens()
"""

import atexit
import logging
import os
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


def initialize_trulens() -> Optional[object]:
    """
    Initialize TruLens monitoring (opt-in only).

    This function should be called ONCE at application startup
    if ENABLE_TRULENS_MONITORING environment variable is set to "1".

    Returns:
        TruLensMonitor instance if successful, None otherwise
    """
    try:
        # Load configuration
        config = _load_config()

        # Check if monitoring is enabled in config
        if not config.get("trulens", {}).get("enabled", False):
            logger.info("TruLens monitoring disabled in config (set trulens.enabled=true)")
            return None

        # Import monitoring components
        from evals.trulens.trulens_wrapper import TruLensMonitor, set_monitor
        from evals.trulens.feedback_functions import create_feedback_functions

        # Build database URL
        database_url = _build_database_url(config)

        # Create feedback functions
        feedback_funcs = create_feedback_functions(config)
        feedback_list = feedback_funcs.get_feedback_functions()

        # Initialize monitor
        sampling_rate = config.get("trulens", {}).get("sampling_rate", 1.0)

        monitor = TruLensMonitor(
            enabled=True,
            database_url=database_url,
            sampling_rate=sampling_rate,
            feedback_functions=feedback_list,
        )

        if not monitor.enabled:
            logger.warning("TruLens monitor failed to initialize")
            return None

        # Instrument GraphRAG instance
        from rag.graph_rag import graph_rag

        monitor.instrument_graph_rag(graph_rag)

        # Store global reference
        set_monitor(monitor)

        # Register cleanup handler
        atexit.register(_cleanup_monitor, monitor)

        logger.info("✅ TruLens continuous monitoring initialized successfully")
        logger.info(f"   Database: {database_url}")
        logger.info(f"   Sampling rate: {sampling_rate * 100}%")
        logger.info(f"   Feedback functions: {len(feedback_list)}")

        return monitor

    except ImportError as e:
        logger.error(
            f"TruLens dependencies not installed: {e}. "
            "Install with: pip install trulens trulens-providers-openai"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to initialize TruLens: {e}", exc_info=True)
        return None


def _load_config() -> dict:
    """
    Load TruLens configuration from YAML file.

    Returns:
        dict: Configuration dictionary
    """
    # Try config.yaml first, fall back to config.example.yaml
    config_paths = [
        Path("evals/trulens/config.yaml"),
        Path("evals/trulens/config.example.yaml"),
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with config_path.open() as f:
                    config = yaml.safe_load(f)
                logger.debug(f"Loaded TruLens config from {config_path}")
                return config
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {e}")

    # Return default minimal config if no file found
    logger.warning("No TruLens config file found, using defaults")
    return {
        "trulens": {
            "enabled": False,
            "sampling_rate": 1.0,
            "database": {
                "backend": "sqlite",
                "sqlite": {"path": "evals/trulens/trulens.db"}
            }
        }
    }


def _build_database_url(config: dict) -> str:
    """
    Build SQLAlchemy database URL from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        str: Database URL (e.g., "postgresql://user:pass@host:port/db")
    """
    db_config = config.get("trulens", {}).get("database", {})
    backend = db_config.get("backend", "sqlite")

    if backend == "postgresql":
        # Build PostgreSQL URL
        pg_config = db_config.get("postgresql", {})

        # Support environment variable expansion
        host = os.path.expandvars(pg_config.get("host", "localhost"))
        port = pg_config.get("port", 5432)
        database = os.path.expandvars(pg_config.get("database", "trulens"))
        user = os.path.expandvars(pg_config.get("user", "postgres"))
        password = os.path.expandvars(pg_config.get("password", ""))

        # Handle missing password
        if not password:
            password_env = os.getenv("TRULENS_DB_PASSWORD")
            if password_env:
                password = password_env
            else:
                logger.warning(
                    "PostgreSQL password not configured. "
                    "Set TRULENS_DB_PASSWORD or trulens.database.postgresql.password"
                )

        database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        logger.debug(f"Using PostgreSQL database: {host}:{port}/{database}")

    else:
        # Fall back to SQLite
        sqlite_config = db_config.get("sqlite", {})
        path = sqlite_config.get("path", "evals/trulens/trulens.db")

        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        database_url = f"sqlite:///{path}"

        logger.debug(f"Using SQLite database: {path}")

    return database_url


def _cleanup_monitor(monitor):
    """
    Cleanup function called on application shutdown.

    Args:
        monitor: TruLensMonitor instance to clean up
    """
    try:
        if monitor is not None:
            monitor.stop()
            logger.info("TruLens monitor stopped")
    except Exception as e:
        logger.warning(f"Error stopping TruLens monitor: {e}")
