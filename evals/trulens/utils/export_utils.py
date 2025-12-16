"""
Export utilities for TruLens monitoring data.

Exports TruLens records to various formats:
- JSON (programmatic access)
- CSV (spreadsheet analysis)
- Markdown (human-readable reports)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TruLensExporter:
    """Export TruLens monitoring data to various formats."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize exporter.

        Args:
            database_url: TruLens database URL (default: from config)
        """
        self.database_url = database_url or "sqlite:///evals/trulens/trulens.db"
        self.tru = None

        try:
            from trulens_eval import Tru
            self.tru = Tru(database_url=self.database_url)
        except Exception as e:
            logger.error(f"Failed to initialize TruLens: {e}")

    def export_to_json(self, output_path: str, limit: int = 1000):
        """
        Export records to JSON.

        Args:
            output_path: Output file path
            limit: Maximum records to export
        """
        if self.tru is None:
            raise RuntimeError("TruLens not initialized")

        try:
            records_df = self.tru.get_records_and_feedback(limit=limit)

            if records_df is None or len(records_df) == 0:
                logger.warning("No records to export")
                return

            # Convert to dict and export
            records_dict = records_df.to_dict('records')

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(records_dict, f, indent=2, default=str)

            logger.info(f"Exported {len(records_dict)} records to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            raise

    def export_to_csv(self, output_path: str, limit: int = 1000):
        """
        Export aggregated metrics to CSV.

        Args:
            output_path: Output file path
            limit: Maximum records to include
        """
        if self.tru is None:
            raise RuntimeError("TruLens not initialized")

        try:
            leaderboard = self.tru.get_leaderboard(limit=limit)

            if leaderboard is None or len(leaderboard) == 0:
                logger.warning("No leaderboard data to export")
                return

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            leaderboard.to_csv(output_path, index=False)

            logger.info(f"Exported leaderboard to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise

    def export_summary_markdown(self, output_path: str):
        """
        Export summary report to Markdown.

        Args:
            output_path: Output file path
        """
        if self.tru is None:
            raise RuntimeError("TruLens not initialized")

        try:
            leaderboard = self.tru.get_leaderboard()

            lines = [
                "# TruLens Monitoring Summary",
                f"*Generated: {datetime.utcnow().isoformat()}*",
                "",
                "## Overview",
                f"- **Total Applications**: {len(leaderboard) if leaderboard is not None else 0}",
                "",
                "## Application Performance",
                "",
            ]

            if leaderboard is not None and len(leaderboard) > 0:
                # Add leaderboard table
                lines.append("| App ID | Latency (s) | Answer Relevance | Groundedness | Context Relevance |")
                lines.append("|--------|-------------|------------------|--------------|-------------------|")

                for _, row in leaderboard.iterrows():
                    app_id = row.get('app_id', 'unknown')
                    latency = row.get('latency', 0.0)
                    answer_rel = row.get('Answer Relevance', 0.0)
                    groundedness = row.get('Groundedness', 0.0)
                    context_rel = row.get('Context Relevance', 0.0)

                    lines.append(
                        f"| {app_id} | "
                        f"{latency:.2f} | "
                        f"{answer_rel:.2f} | "
                        f"{groundedness:.2f} | "
                        f"{context_rel:.2f} |"
                    )
            else:
                lines.append("*No data available*")

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write("\n".join(lines))

            logger.info(f"Exported summary to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export summary: {e}")
            raise
