"""
LLM Usage Tracker - SQLite-based token usage tracking.

Tracks all LLM calls across the application with per-document,
per-query granularity for analytics and cost monitoring.
"""

import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "llm_usage.db")


class LLMUsageTracker:
    """Thread-safe SQLite-based LLM usage tracker."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._local = threading.local()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            self._local.connection = sqlite3.connect(DB_PATH, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_database(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                -- Context
                operation TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                
                -- Tokens
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                
                -- Granularity
                document_id TEXT,
                conversation_id TEXT,
                query_id TEXT,
                
                -- Metadata
                success INTEGER DEFAULT 1,
                latency_ms INTEGER,
                error_message TEXT
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON llm_usage(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_operation ON llm_usage(operation)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_document ON llm_usage(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation ON llm_usage(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_provider ON llm_usage(provider)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON llm_usage(model)")

        conn.commit()
        logger.info(f"LLM usage database initialized at {DB_PATH}")

    def record(
        self,
        operation: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        document_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        query_id: Optional[str] = None,
        success: bool = True,
        latency_ms: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> int:
        """
        Record an LLM usage event.

        Args:
            operation: Type of operation (e.g., 'ingestion.entity_extraction')
            provider: LLM provider (e.g., 'openai', 'anthropic')
            model: Model name (e.g., 'gpt-4o-mini')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            document_id: Optional document ID for ingestion ops
            conversation_id: Optional conversation ID for chat ops
            query_id: Optional unique query ID
            success: Whether the call was successful
            latency_ms: Optional latency in milliseconds
            error_message: Optional error message if failed

        Returns:
            ID of the inserted record
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO llm_usage (
                    operation, provider, model, input_tokens, output_tokens,
                    document_id, conversation_id, query_id,
                    success, latency_ms, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    operation,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    document_id,
                    conversation_id,
                    query_id or str(uuid.uuid4()),
                    1 if success else 0,
                    latency_ms,
                    error_message,
                ),
            )
            conn.commit()
            return cursor.lastrowid

        except Exception as e:
            logger.error(f"Failed to record LLM usage: {e}")
            return -1

    @contextmanager
    def track(
        self,
        operation: str,
        provider: str,
        model: str,
        document_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ):
        """
        Context manager for tracking LLM calls with automatic timing.

        Usage:
            with tracker.track("rag.generation", "openai", "gpt-4o-mini") as ctx:
                result = llm.generate_response(..., include_usage=True)
                ctx.set_usage(result["usage"]["input"], result["usage"]["output"])
        """
        ctx = _TrackingContext(
            tracker=self,
            operation=operation,
            provider=provider,
            model=model,
            document_id=document_id,
            conversation_id=conversation_id,
        )
        ctx.start()
        try:
            yield ctx
        except Exception as e:
            ctx.set_error(str(e))
            raise
        finally:
            ctx.finish()

    def get_summary(self) -> Dict[str, Any]:
        """Get overall usage summary."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                COUNT(*) as total_calls,
                SUM(input_tokens) as total_input,
                SUM(output_tokens) as total_output,
                SUM(input_tokens + output_tokens) as total_tokens,
                AVG(latency_ms) as avg_latency_ms
            FROM llm_usage
            WHERE success = 1
        """)
        row = cursor.fetchone()

        return {
            "total_calls": row["total_calls"] or 0,
            "total_input_tokens": row["total_input"] or 0,
            "total_output_tokens": row["total_output"] or 0,
            "total_tokens": row["total_tokens"] or 0,
            "avg_latency_ms": round(row["avg_latency_ms"] or 0, 2),
        }

    def get_by_operation(self) -> Dict[str, Dict[str, int]]:
        """Get usage breakdown by operation."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                operation,
                COUNT(*) as calls,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens
            FROM llm_usage
            WHERE success = 1
            GROUP BY operation
            ORDER BY SUM(input_tokens + output_tokens) DESC
        """)

        return {
            row["operation"]: {
                "calls": row["calls"],
                "input": row["input_tokens"] or 0,
                "output": row["output_tokens"] or 0,
            }
            for row in cursor.fetchall()
        }

    def get_by_provider(self) -> Dict[str, Dict[str, int]]:
        """Get usage breakdown by provider."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                provider,
                COUNT(*) as calls,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens
            FROM llm_usage
            WHERE success = 1
            GROUP BY provider
        """)

        return {
            row["provider"]: {
                "calls": row["calls"],
                "input": row["input_tokens"] or 0,
                "output": row["output_tokens"] or 0,
            }
            for row in cursor.fetchall()
        }

    def get_by_model(self) -> Dict[str, Dict[str, int]]:
        """Get usage breakdown by model."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                model,
                COUNT(*) as calls,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens
            FROM llm_usage
            WHERE success = 1
            GROUP BY model
            ORDER BY SUM(input_tokens + output_tokens) DESC
        """)

        return {
            row["model"]: {
                "calls": row["calls"],
                "input": row["input_tokens"] or 0,
                "output": row["output_tokens"] or 0,
            }
            for row in cursor.fetchall()
        }

    def get_by_document(self, document_id: str) -> Dict[str, Any]:
        """Get usage for a specific document."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                COUNT(*) as calls,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                MIN(timestamp) as first_call,
                MAX(timestamp) as last_call
            FROM llm_usage
            WHERE document_id = ? AND success = 1
            """,
            (document_id,),
        )
        row = cursor.fetchone()

        return {
            "document_id": document_id,
            "calls": row["calls"] or 0,
            "input_tokens": row["input_tokens"] or 0,
            "output_tokens": row["output_tokens"] or 0,
            "total_tokens": (row["input_tokens"] or 0) + (row["output_tokens"] or 0),
            "first_call": row["first_call"],
            "last_call": row["last_call"],
        }

    def get_by_conversation(self, limit: int = 20) -> list:
        """Get usage breakdown by conversation_id with cost estimates in EUR."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Pricing per 1M tokens (mid 2024 rates)
        input_rate = 0.15 / 1_000_000  # gpt-4o-mini default
        output_rate = 0.60 / 1_000_000
        usd_to_eur = 0.92

        cursor.execute(
            """
            SELECT 
                conversation_id,
                COUNT(*) as calls,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                MIN(timestamp) as first_call,
                MAX(timestamp) as last_call
            FROM llm_usage
            WHERE conversation_id IS NOT NULL AND conversation_id != '' AND success = 1
            GROUP BY conversation_id
            ORDER BY (SUM(input_tokens) + SUM(output_tokens)) DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

        result = []
        for row in rows:
            input_tokens = row["input_tokens"] or 0
            output_tokens = row["output_tokens"] or 0
            cost_usd = (input_tokens * input_rate) + (output_tokens * output_rate)
            cost_eur = cost_usd * usd_to_eur

            result.append({
                "conversation_id": row["conversation_id"],
                "calls": row["calls"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_eur": round(cost_eur, 4),
                "first_call": row["first_call"],
                "last_call": row["last_call"],
            })

        return result

    def get_recent(self, limit: int = 100) -> list:
        """Get recent usage records."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM llm_usage
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_cost_estimate(self, custom_pricing: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Get estimated costs based on token usage.
        
        Default pricing is per 1M tokens (as of Dec 2024):
        - gpt-4o-mini: $0.15 input, $0.60 output
        - gpt-4o: $2.50 input, $10.00 output
        - claude-3-5-sonnet: $3.00 input, $15.00 output
        - claude-3-5-haiku: $0.80 input, $4.00 output
        """
        # Default pricing per 1M tokens
        default_pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
            "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
            "claude-sonnet-4-5-20250514": {"input": 3.00, "output": 15.00},
            "mistral-large-latest": {"input": 2.00, "output": 6.00},
            "mistral-small-latest": {"input": 0.20, "output": 0.60},
            # Default fallback for unknown models
            "_default": {"input": 1.00, "output": 3.00},
        }
        
        # USD to EUR exchange rate (approximate, can be overridden via env var)
        import os
        usd_to_eur = float(os.environ.get("USD_TO_EUR_RATE", "0.96"))
        
        pricing = {**default_pricing, **(custom_pricing or {})}
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                model,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens
            FROM llm_usage
            WHERE success = 1
            GROUP BY model
        """)
        
        total_cost = 0.0
        by_model = {}
        
        for row in cursor.fetchall():
            model = row["model"]
            input_tokens = row["input_tokens"] or 0
            output_tokens = row["output_tokens"] or 0
            
            # Get pricing for this model (fallback to default)
            model_pricing = pricing.get(model, pricing.get("_default", {"input": 1.0, "output": 3.0}))
            
            # Calculate cost (price is per 1M tokens)
            input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
            output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
            model_cost = input_cost + output_cost
            
            by_model[model] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
                "total_cost": round(model_cost, 4),
                "total_cost_eur": round(model_cost * usd_to_eur, 4),
            }
            total_cost += model_cost
        
        return {
            "total_cost_usd": round(total_cost, 4),
            "total_cost_eur": round(total_cost * usd_to_eur, 4),
            "usd_to_eur_rate": usd_to_eur,
            "by_model": by_model,
        }
    
    def get_time_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get usage trends over time (daily aggregates)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                date(timestamp) as date,
                COUNT(*) as calls,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(input_tokens + output_tokens) as total_tokens,
                AVG(latency_ms) as avg_latency_ms
            FROM llm_usage
            WHERE success = 1 
              AND timestamp >= datetime('now', ?)
            GROUP BY date(timestamp)
            ORDER BY date(timestamp) ASC
        """, (f'-{days} days',))
        
        daily = []
        for row in cursor.fetchall():
            daily.append({
                "date": row["date"],
                "calls": row["calls"],
                "input_tokens": row["input_tokens"] or 0,
                "output_tokens": row["output_tokens"] or 0,
                "total_tokens": row["total_tokens"] or 0,
                "avg_latency_ms": round(row["avg_latency_ms"] or 0, 2),
            })
        
        # Also get hourly for last 24h
        cursor.execute("""
            SELECT 
                strftime('%Y-%m-%d %H:00', timestamp) as hour,
                COUNT(*) as calls,
                SUM(input_tokens + output_tokens) as total_tokens
            FROM llm_usage
            WHERE success = 1 
              AND timestamp >= datetime('now', '-24 hours')
            GROUP BY strftime('%Y-%m-%d %H:00', timestamp)
            ORDER BY hour ASC
        """)
        
        hourly = []
        for row in cursor.fetchall():
            hourly.append({
                "hour": row["hour"],
                "calls": row["calls"],
                "total_tokens": row["total_tokens"] or 0,
            })
        
        return {
            "daily": daily,
            "hourly": hourly,
        }
    
    def get_success_rate(self) -> Dict[str, Any]:
        """Get success/error rate statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed
            FROM llm_usage
        """)
        
        row = cursor.fetchone()
        total = row["total"] or 0
        successful = row["successful"] or 0
        failed = row["failed"] or 0
        
        success_rate = (successful / total * 100) if total > 0 else 100.0
        
        # Get recent errors
        cursor.execute("""
            SELECT 
                timestamp,
                operation,
                model,
                error_message
            FROM llm_usage
            WHERE success = 0
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        
        recent_errors = [dict(row) for row in cursor.fetchall()]
        
        return {
            "total_calls": total,
            "successful": successful,
            "failed": failed,
            "success_rate": round(success_rate, 2),
            "recent_errors": recent_errors,
        }
    
    def get_efficiency_metrics(self) -> Dict[str, Any]:
        """Get token efficiency metrics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Overall input/output ratio
        cursor.execute("""
            SELECT 
                SUM(input_tokens) as total_input,
                SUM(output_tokens) as total_output,
                AVG(input_tokens) as avg_input,
                AVG(output_tokens) as avg_output
            FROM llm_usage
            WHERE success = 1
        """)
        
        row = cursor.fetchone()
        total_input = row["total_input"] or 0
        total_output = row["total_output"] or 0
        
        io_ratio = (total_output / total_input) if total_input > 0 else 0
        
        # Per-operation efficiency
        cursor.execute("""
            SELECT 
                operation,
                COUNT(*) as calls,
                AVG(input_tokens) as avg_input,
                AVG(output_tokens) as avg_output,
                AVG(input_tokens + output_tokens) as avg_total
            FROM llm_usage
            WHERE success = 1
            GROUP BY operation
            ORDER BY AVG(input_tokens + output_tokens) DESC
        """)
        
        by_operation = {}
        for r in cursor.fetchall():
            by_operation[r["operation"]] = {
                "calls": r["calls"],
                "avg_input": round(r["avg_input"] or 0, 1),
                "avg_output": round(r["avg_output"] or 0, 1),
                "avg_total": round(r["avg_total"] or 0, 1),
            }
        
        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "io_ratio": round(io_ratio, 3),
            "avg_input_per_call": round(row["avg_input"] or 0, 1),
            "avg_output_per_call": round(row["avg_output"] or 0, 1),
            "by_operation": by_operation,
        }


class _TrackingContext:
    """Internal context for tracking LLM calls."""

    def __init__(
        self,
        tracker: LLMUsageTracker,
        operation: str,
        provider: str,
        model: str,
        document_id: Optional[str],
        conversation_id: Optional[str],
    ):
        self.tracker = tracker
        self.operation = operation
        self.provider = provider
        self.model = model
        self.document_id = document_id
        self.conversation_id = conversation_id
        self.query_id = str(uuid.uuid4())
        self.input_tokens = 0
        self.output_tokens = 0
        self.start_time = 0
        self.success = True
        self.error_message = None

    def start(self):
        self.start_time = time.time()

    def set_usage(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def set_error(self, message: str):
        self.success = False
        self.error_message = message

    def finish(self):
        latency_ms = int((time.time() - self.start_time) * 1000)
        self.tracker.record(
            operation=self.operation,
            provider=self.provider,
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            document_id=self.document_id,
            conversation_id=self.conversation_id,
            query_id=self.query_id,
            success=self.success,
            latency_ms=latency_ms,
            error_message=self.error_message,
        )


# Singleton instance
usage_tracker = LLMUsageTracker()
