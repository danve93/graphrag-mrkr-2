"""
TruLens dashboard launcher.

Launches the TruLens Streamlit dashboard for viewing monitoring data.

Updated for TruLens v2.5+ API.

Usage:
    python evals/trulens/dashboard_launcher.py --db postgresql://user:pass@host:port/db
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def launch_dashboard(
    db_path: str = "evals/trulens/trulens.db",
    host: str = "localhost",
    port: int = 8501,
):
    """
    Launch TruLens Streamlit dashboard.

    Args:
        db_path: Path to TruLens database or PostgreSQL URL
        host: Dashboard host
        port: Dashboard port
    """
    try:
        from trulens.core import TruSession
        from trulens.dashboard import run_dashboard

        # Determine database URL
        if db_path.startswith("postgresql://"):
            database_url = db_path
        else:
            database_url = f"sqlite:///{db_path}"

        # Initialize TruLens session with database
        session = TruSession(database_url=database_url)

        # NOTE: Do NOT reset database on dashboard launch
        # This was clearing all records every time!
        # The database schema is created automatically by TruSession

        print(f"🚀 Launching TruLens dashboard...")
        print(f"   Database: {database_url}")
        print(f"   Dashboard: http://{host}:{port}")
        print(f"\nPress Ctrl+C to stop\n")

        # Launch dashboard (new v2.5+ API)
        run_dashboard(
            session,
            port=port,
            force=True,  # Force launch even if port is in use
        )

    except ImportError as e:
        print(f"❌ TruLens not installed: {e}")
        print("Install with: pip install trulens trulens-providers-openai")
    except Exception as e:
        logger.error(f"Failed to launch dashboard: {e}", exc_info=True)
        print(f"❌ Failed to launch dashboard: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch TruLens monitoring dashboard")
    parser.add_argument(
        "--db",
        default="evals/trulens/trulens.db",
        help="Path to TruLens database (SQLite path or PostgreSQL URL)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Dashboard host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Dashboard port (default: 8501)"
    )

    args = parser.parse_args()
    launch_dashboard(args.db, args.host, args.port)
