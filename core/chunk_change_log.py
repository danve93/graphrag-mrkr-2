"""
Chunk Change Log - SQLite storage for tracking chunk edit history.

Provides audit trail for chunk modifications (edit, delete, merge) to enable:
- Pattern learning from user corrections
- Undo/history browsing
- JSON export for analysis
"""

import sqlite3
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Database location
DEFAULT_DB_PATH = Path("data/chunk_changes.db")


class ChunkChangeLog:
    """SQLite-based storage for chunk change history."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    before_content TEXT,
                    before_hash TEXT,
                    after_content TEXT,
                    after_hash TEXT,
                    reasoning TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_changes_doc_id 
                ON chunk_changes(document_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_changes_chunk_id 
                ON chunk_changes(chunk_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_changes_action 
                ON chunk_changes(action)
            """)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _hash_content(content: Optional[str]) -> Optional[str]:
        """Generate SHA-256 hash of content for comparison."""
        if content is None:
            return None
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _truncate_content(content: Optional[str], max_length: int = 500) -> Optional[str]:
        """Truncate content for storage efficiency."""
        if content is None:
            return None
        if len(content) <= max_length:
            return content
        return content[:max_length] + f"... [truncated, {len(content)} chars total]"

    def log_change(
        self,
        document_id: str,
        chunk_id: str,
        action: str,
        before_content: Optional[str] = None,
        after_content: Optional[str] = None,
        reasoning: Optional[str] = None,
        metadata: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> int:
        """
        Log a chunk change to the database.

        Args:
            document_id: Document the chunk belongs to
            chunk_id: ID of the modified chunk
            action: Type of change (edit, delete, merge, split)
            before_content: Content before the change
            after_content: Content after the change
            reasoning: User-provided or AI-generated reason for change
            metadata: Additional metadata (merged_chunk_ids, etc.)
            error: Error message if the change failed

        Returns:
            ID of the created change record
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chunk_changes (
                    document_id, chunk_id, action,
                    before_content, before_hash,
                    after_content, after_hash,
                    reasoning, metadata, created_at, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk_id,
                    action,
                    self._truncate_content(before_content),
                    self._hash_content(before_content),
                    self._truncate_content(after_content),
                    self._hash_content(after_content),
                    reasoning,
                    json.dumps(metadata) if metadata else None,
                    datetime.utcnow().isoformat(),
                    error,
                ),
            )
            conn.commit()
            change_id = cursor.lastrowid
            logger.info(
                "Logged chunk change: doc=%s chunk=%s action=%s id=%d",
                document_id,
                chunk_id,
                action,
                change_id,
            )
            return change_id

    def get_changes(
        self,
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve change history with optional filters.

        Args:
            document_id: Filter by document
            chunk_id: Filter by chunk
            action: Filter by action type
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of change records
        """
        conditions = []
        params = []

        if document_id:
            conditions.append("document_id = ?")
            params.append(document_id)
        if chunk_id:
            conditions.append("chunk_id = ?")
            params.append(chunk_id)
        if action:
            conditions.append("action = ?")
            params.append(action)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM chunk_changes
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()

            return [self._row_to_dict(row) for row in rows]

    def get_changes_count(
        self,
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        action: Optional[str] = None,
    ) -> int:
        """Count changes matching filters."""
        conditions = []
        params = []

        if document_id:
            conditions.append("document_id = ?")
            params.append(document_id)
        if chunk_id:
            conditions.append("chunk_id = ?")
            params.append(chunk_id)
        if action:
            conditions.append("action = ?")
            params.append(action)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_connection() as conn:
            result = conn.execute(
                f"SELECT COUNT(*) FROM chunk_changes WHERE {where_clause}",
                params,
            ).fetchone()
            return result[0]

    def export_changes(
        self,
        document_id: Optional[str] = None,
        include_content: bool = True,
    ) -> dict[str, Any]:
        """
        Export changes as JSON-serializable dict.

        Args:
            document_id: Filter by document (None for all)
            include_content: Whether to include before/after content

        Returns:
            Export data with metadata and changes
        """
        changes = self.get_changes(document_id=document_id, limit=10000)

        if not include_content:
            for change in changes:
                change.pop("before_content", None)
                change.pop("after_content", None)

        return {
            "export_date": datetime.utcnow().isoformat(),
            "document_id": document_id,
            "total_changes": len(changes),
            "changes": changes,
        }

    def get_action_summary(self, document_id: Optional[str] = None) -> dict[str, int]:
        """Get count of changes grouped by action type."""
        params = [document_id] if document_id else []
        where_clause = "WHERE document_id = ?" if document_id else ""

        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT action, COUNT(*) as count
                FROM chunk_changes
                {where_clause}
                GROUP BY action
                """,
                params,
            ).fetchall()

            return {row["action"]: row["count"] for row in rows}

    def delete_changes(
        self,
        document_id: Optional[str] = None,
        before_date: Optional[str] = None,
    ) -> int:
        """
        Delete change records.

        Args:
            document_id: Delete changes for specific document
            before_date: Delete changes older than date (ISO format)

        Returns:
            Number of deleted records
        """
        conditions = []
        params = []

        if document_id:
            conditions.append("document_id = ?")
            params.append(document_id)
        if before_date:
            conditions.append("created_at < ?")
            params.append(before_date)

        if not conditions:
            raise ValueError("Must specify document_id or before_date to delete")

        where_clause = " AND ".join(conditions)

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM chunk_changes WHERE {where_clause}",
                params,
            )
            conn.commit()
            deleted = cursor.rowcount
            logger.info("Deleted %d chunk change records", deleted)
            return deleted

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert sqlite row to dict, parsing JSON metadata."""
        d = dict(row)
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except json.JSONDecodeError:
                pass
        return d


# Global instance
_change_log: Optional[ChunkChangeLog] = None


def get_change_log() -> ChunkChangeLog:
    """Get or create global ChunkChangeLog instance."""
    global _change_log
    if _change_log is None:
        _change_log = ChunkChangeLog()
    return _change_log
