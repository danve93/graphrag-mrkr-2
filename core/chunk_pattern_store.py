"""
Chunk Pattern Store - SQLite storage for user-defined chunk patterns.

Patterns can be:
- Pre-seeded (built-in heuristics)
- Learned from user corrections
- Manually defined by user

Each pattern includes:
- Match criteria (regex, length threshold, etc.)
- Suggested action (delete, merge, edit)
- Confidence score
- Usage statistics
"""

import sqlite3
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Database location
DEFAULT_DB_PATH = Path("data/chunk_patterns.db")


@dataclass
class ChunkPattern:
    """A chunk pattern definition."""
    id: str
    name: str
    description: str
    match_type: str  # 'regex', 'length', 'content', 'similarity'
    match_criteria: dict  # type-specific criteria
    action: str  # 'delete', 'merge', 'edit', 'flag'
    confidence: float
    is_builtin: bool = False
    enabled: bool = True
    usage_count: int = 0
    last_used: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "match_type": self.match_type,
            "match_criteria": self.match_criteria,
            "action": self.action,
            "confidence": self.confidence,
            "is_builtin": self.is_builtin,
            "enabled": self.enabled,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkPattern":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            match_type=data["match_type"],
            match_criteria=data.get("match_criteria", {}),
            action=data["action"],
            confidence=data.get("confidence", 0.7),
            is_builtin=data.get("is_builtin", False),
            enabled=data.get("enabled", True),
            usage_count=data.get("usage_count", 0),
            last_used=data.get("last_used"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


# Built-in patterns
BUILTIN_PATTERNS = [
    ChunkPattern(
        id="builtin-short-chunk",
        name="Very Short Chunks",
        description="Chunks with fewer than 20 characters are likely noise",
        match_type="length",
        match_criteria={"max_length": 20},
        action="delete",
        confidence=0.8,
        is_builtin=True,
    ),
    ChunkPattern(
        id="builtin-placeholder",
        name="Placeholder Text",
        description="Common placeholder patterns like TODO, [TBD], etc.",
        match_type="regex",
        match_criteria={"pattern": r"^\s*(TODO|TBD|\[.*\]|<.*>|placeholder)\s*$", "flags": "i"},
        action="delete",
        confidence=0.85,
        is_builtin=True,
    ),
    ChunkPattern(
        id="builtin-separator",
        name="Separator Lines",
        description="Lines containing only separator characters",
        match_type="regex",
        match_criteria={"pattern": r"^[\s\-=_*#]{3,}$"},
        action="delete",
        confidence=0.9,
        is_builtin=True,
    ),
    ChunkPattern(
        id="builtin-low-density",
        name="Low Information Density",
        description="Chunks with mostly whitespace or punctuation",
        match_type="content",
        match_criteria={"min_alpha_ratio": 0.3},
        action="delete",
        confidence=0.7,
        is_builtin=True,
    ),
    ChunkPattern(
        id="builtin-consecutive-short",
        name="Consecutive Short Chunks",
        description="Multiple consecutive short chunks that could be merged",
        match_type="length",
        match_criteria={"max_length": 50, "min_consecutive": 2},
        action="merge",
        confidence=0.75,
        is_builtin=True,
    ),
    ChunkPattern(
        id="builtin-duplicate-content",
        name="Duplicate Content",
        description="Chunks with identical or near-identical content",
        match_type="similarity",
        match_criteria={"method": "exact_normalized"},
        action="delete",
        confidence=0.9,
        is_builtin=True,
    ),
]


class ChunkPatternStore:
    """SQLite-based storage for chunk patterns."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._seed_builtin_patterns()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_patterns (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    match_type TEXT NOT NULL,
                    match_criteria TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL DEFAULT 0.7,
                    is_builtin INTEGER DEFAULT 0,
                    enabled INTEGER DEFAULT 1,
                    usage_count INTEGER DEFAULT 0,
                    last_used TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_action 
                ON chunk_patterns(action)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_enabled 
                ON chunk_patterns(enabled)
            """)
            conn.commit()

    def _seed_builtin_patterns(self) -> None:
        """Seed built-in patterns if not already present."""
        with self._get_connection() as conn:
            for pattern in BUILTIN_PATTERNS:
                existing = conn.execute(
                    "SELECT id FROM chunk_patterns WHERE id = ?",
                    (pattern.id,)
                ).fetchone()
                if not existing:
                    self._insert_pattern(conn, pattern)
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

    def _insert_pattern(self, conn, pattern: ChunkPattern) -> None:
        """Insert a pattern into the database."""
        now = datetime.utcnow().isoformat()
        conn.execute(
            """
            INSERT INTO chunk_patterns (
                id, name, description, match_type, match_criteria,
                action, confidence, is_builtin, enabled, usage_count,
                last_used, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pattern.id,
                pattern.name,
                pattern.description,
                pattern.match_type,
                json.dumps(pattern.match_criteria),
                pattern.action,
                pattern.confidence,
                1 if pattern.is_builtin else 0,
                1 if pattern.enabled else 0,
                pattern.usage_count,
                pattern.last_used,
                pattern.created_at or now,
                pattern.updated_at or now,
            ),
        )

    def get_patterns(
        self,
        enabled_only: bool = False,
        action: Optional[str] = None,
        include_builtin: bool = True,
    ) -> list[ChunkPattern]:
        """Get all patterns with optional filters."""
        conditions = []
        params = []

        if enabled_only:
            conditions.append("enabled = 1")
        if action:
            conditions.append("action = ?")
            params.append(action)
        if not include_builtin:
            conditions.append("is_builtin = 0")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM chunk_patterns
                WHERE {where_clause}
                ORDER BY is_builtin DESC, usage_count DESC, name ASC
                """,
                params,
            ).fetchall()

            return [self._row_to_pattern(row) for row in rows]

    def get_pattern(self, pattern_id: str) -> Optional[ChunkPattern]:
        """Get a single pattern by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM chunk_patterns WHERE id = ?",
                (pattern_id,),
            ).fetchone()
            return self._row_to_pattern(row) if row else None

    def create_pattern(self, pattern: ChunkPattern) -> ChunkPattern:
        """Create a new pattern."""
        if not pattern.id:
            pattern.id = str(uuid.uuid4())
        pattern.created_at = datetime.utcnow().isoformat()
        pattern.updated_at = pattern.created_at

        with self._get_connection() as conn:
            self._insert_pattern(conn, pattern)
            conn.commit()
            logger.info("Created pattern: %s (%s)", pattern.name, pattern.id)
            return pattern

    def update_pattern(
        self, pattern_id: str, updates: dict[str, Any]
    ) -> Optional[ChunkPattern]:
        """Update an existing pattern."""
        pattern = self.get_pattern(pattern_id)
        if not pattern:
            return None

        # Don't allow changing is_builtin
        updates.pop("is_builtin", None)
        updates.pop("id", None)
        updates["updated_at"] = datetime.utcnow().isoformat()

        # Convert match_criteria to JSON if present
        if "match_criteria" in updates and isinstance(updates["match_criteria"], dict):
            updates["match_criteria"] = json.dumps(updates["match_criteria"])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [pattern_id]

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE chunk_patterns SET {set_clause} WHERE id = ?",
                values,
            )
            conn.commit()
            logger.info("Updated pattern: %s", pattern_id)
            return self.get_pattern(pattern_id)

    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern (built-in patterns cannot be deleted)."""
        pattern = self.get_pattern(pattern_id)
        if not pattern:
            return False
        if pattern.is_builtin:
            logger.warning("Cannot delete built-in pattern: %s", pattern_id)
            return False

        with self._get_connection() as conn:
            conn.execute("DELETE FROM chunk_patterns WHERE id = ?", (pattern_id,))
            conn.commit()
            logger.info("Deleted pattern: %s", pattern_id)
            return True

    def increment_usage(self, pattern_id: str) -> None:
        """Increment usage count for a pattern."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE chunk_patterns 
                SET usage_count = usage_count + 1, last_used = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), pattern_id),
            )
            conn.commit()

    def toggle_pattern(self, pattern_id: str, enabled: bool) -> Optional[ChunkPattern]:
        """Enable or disable a pattern."""
        return self.update_pattern(pattern_id, {"enabled": 1 if enabled else 0})

    def export_patterns(self, include_builtin: bool = False) -> dict[str, Any]:
        """Export patterns as JSON."""
        patterns = self.get_patterns(include_builtin=include_builtin)
        return {
            "export_date": datetime.utcnow().isoformat(),
            "version": "1.0",
            "total_patterns": len(patterns),
            "patterns": [p.to_dict() for p in patterns],
        }

    def import_patterns(
        self, data: dict[str, Any], overwrite: bool = False
    ) -> dict[str, int]:
        """Import patterns from JSON export."""
        results = {"created": 0, "updated": 0, "skipped": 0}

        patterns_data = data.get("patterns", [])
        for pdata in patterns_data:
            # Skip built-in patterns
            if pdata.get("is_builtin"):
                results["skipped"] += 1
                continue

            existing = self.get_pattern(pdata.get("id", ""))
            if existing:
                if overwrite:
                    self.update_pattern(existing.id, pdata)
                    results["updated"] += 1
                else:
                    results["skipped"] += 1
            else:
                pattern = ChunkPattern.from_dict(pdata)
                pattern.id = str(uuid.uuid4())  # New ID on import
                self.create_pattern(pattern)
                results["created"] += 1

        logger.info("Imported patterns: %s", results)
        return results

    @staticmethod
    def _row_to_pattern(row: sqlite3.Row) -> ChunkPattern:
        """Convert sqlite row to ChunkPattern."""
        match_criteria = row["match_criteria"]
        if isinstance(match_criteria, str):
            try:
                match_criteria = json.loads(match_criteria)
            except json.JSONDecodeError:
                match_criteria = {}

        return ChunkPattern(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            match_type=row["match_type"],
            match_criteria=match_criteria,
            action=row["action"],
            confidence=row["confidence"],
            is_builtin=bool(row["is_builtin"]),
            enabled=bool(row["enabled"]),
            usage_count=row["usage_count"],
            last_used=row["last_used"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# Global instance
_pattern_store: Optional[ChunkPatternStore] = None


def get_pattern_store() -> ChunkPatternStore:
    """Get or create global ChunkPatternStore instance."""
    global _pattern_store
    if _pattern_store is None:
        _pattern_store = ChunkPatternStore()
    return _pattern_store
