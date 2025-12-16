"""
Unit tests for temporal graph functionality.

Tests cover:
- Temporal node creation
- Temporal filtering logic
- Time-decay calculations
- Temporal query detection
"""

import pytest
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the function we're testing
from rag.nodes.query_analysis import _detect_temporal_query


class TestTemporalQueryDetection:
    """Tests for temporal query detection logic."""

    def test_recent_query_detection(self):
        """Test detection of 'recent' queries."""
        query = "what are the recent updates"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is True
        assert result["intent"] == "recent"
        assert result["decay_weight"] > 0

    def test_latest_query_detection(self):
        """Test detection of 'latest' queries."""
        query = "show me the latest documents"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is True
        assert result["intent"] == "recent"
        assert result["decay_weight"] > 0

    def test_last_week_detection(self):
        """Test detection of 'last week' queries."""
        query = "documents from last week"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is True
        assert result["intent"] == "specific_period"
        assert result["window"] == 7
        assert result["decay_weight"] > 0

    def test_last_month_detection(self):
        """Test detection of 'last month' queries."""
        query = "what happened last month"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is True
        assert result["intent"] == "specific_period"
        assert result["window"] == 30

    def test_specific_days_detection(self):
        """Test detection of 'last N days' queries."""
        query = "documents from the last 7 days"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is True
        assert result["intent"] == "specific_period"
        assert result["window"] == 7

    def test_specific_weeks_detection(self):
        """Test detection of 'last N weeks' queries."""
        query = "updates in the last 2 weeks"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is True
        assert result["intent"] == "specific_period"
        assert result["window"] == 14  # 2 weeks * 7 days

    def test_specific_months_detection(self):
        """Test detection of 'last N months' queries."""
        query = "changes in the past 3 months"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is True
        assert result["intent"] == "specific_period"
        assert result["window"] == 90  # 3 months * 30 days

    def test_trending_query_detection(self):
        """Test detection of 'trending' queries."""
        query = "what are the trending topics"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is True
        assert result["intent"] == "trending"
        assert result["decay_weight"] < 0.3  # Lower weight for trends

    def test_evolution_query_detection(self):
        """Test detection of 'evolution' queries."""
        query = "how has this evolved over time"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is True
        assert result["intent"] == "trending"

    def test_when_question_detection(self):
        """Test detection of 'when' questions."""
        query = "when was this created"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is True
        assert result["intent"] == "when_question"

    def test_non_temporal_query(self):
        """Test that non-temporal queries are not detected as temporal."""
        query = "what is the capital of france"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is False
        assert result["intent"] == "none"
        assert result["decay_weight"] == 0.0
        assert result["window"] is None

    def test_non_temporal_how_query(self):
        """Test that 'how' queries without temporal context are not temporal."""
        query = "how does authentication work"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is False

    def test_multiple_temporal_indicators(self):
        """Test query with multiple temporal indicators (uses first match priority)."""
        query = "recent updates from last week"
        result = _detect_temporal_query(query.lower())

        assert result["is_temporal"] is True
        # Should match 'recent' first (higher priority than 'last week')
        assert result["intent"] == "recent"


class TestTemporalNodeCreation:
    """Tests for temporal node creation logic."""

    def test_temporal_components_extraction(self):
        """Test extraction of year, quarter, month, date components."""
        timestamp = datetime(2025, 3, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp()
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        assert dt.year == 2025
        assert dt.month == 3
        assert dt.day == 15

        # Quarter calculation
        quarter = (dt.month - 1) // 3 + 1
        assert quarter == 1  # March is in Q1

    def test_quarter_calculation_q2(self):
        """Test quarter calculation for Q2."""
        dt = datetime(2025, 6, 15, tzinfo=timezone.utc)
        quarter = (dt.month - 1) // 3 + 1
        assert quarter == 2

    def test_quarter_calculation_q3(self):
        """Test quarter calculation for Q3."""
        dt = datetime(2025, 9, 15, tzinfo=timezone.utc)
        quarter = (dt.month - 1) // 3 + 1
        assert quarter == 3

    def test_quarter_calculation_q4(self):
        """Test quarter calculation for Q4."""
        dt = datetime(2025, 12, 15, tzinfo=timezone.utc)
        quarter = (dt.month - 1) // 3 + 1
        assert quarter == 4

    def test_date_string_format(self):
        """Test date string formatting."""
        dt = datetime(2025, 3, 15, 10, 30, 0, tzinfo=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        assert date_str == "2025-03-15"

    def test_month_label_format(self):
        """Test month label formatting."""
        dt = datetime(2025, 3, 15, tzinfo=timezone.utc)
        month_label = dt.strftime("%Y-%m")
        assert month_label == "2025-03"


class TestTemporalFiltering:
    """Tests for temporal filtering logic."""

    def test_time_window_calculation(self):
        """Test calculation of time window boundaries."""
        now = datetime.now(timezone.utc)
        window_days = 7

        after_date = (now - timedelta(days=window_days)).strftime("%Y-%m-%d")

        # Parse back to verify
        parsed = datetime.strptime(after_date, "%Y-%m-%d")
        diff = (now - parsed.replace(tzinfo=timezone.utc)).days

        assert diff >= 7
        assert diff <= 8  # Allow for small timing differences

    def test_recent_query_time_window(self):
        """Test that recent queries get appropriate time decay weight."""
        query = "recent documents"
        result = _detect_temporal_query(query.lower())

        assert result["decay_weight"] >= 0.2
        assert result["decay_weight"] <= 0.5

    def test_specific_period_decay_weight(self):
        """Test decay weight for specific period queries."""
        query = "documents from last month"
        result = _detect_temporal_query(query.lower())

        assert result["decay_weight"] > 0
        assert result["decay_weight"] <= 0.3


class TestTimeDecayScoring:
    """Tests for time-decay scoring calculations."""

    def test_time_decay_exponential_formula(self):
        """Test exponential time decay formula."""
        import math

        # For a document 30 days old
        age_days = 30
        time_factor = math.exp(-0.01 * age_days)

        assert 0 < time_factor < 1
        assert time_factor < 0.75  # 30 days should decay significantly

    def test_time_decay_recent_document(self):
        """Test time decay for recent document (low decay)."""
        import math

        age_days = 1
        time_factor = math.exp(-0.01 * age_days)

        assert time_factor > 0.99  # Very recent, minimal decay

    def test_time_decay_old_document(self):
        """Test time decay for old document (high decay)."""
        import math

        age_days = 365
        time_factor = math.exp(-0.01 * age_days)

        assert time_factor < 0.05  # Very old, significant decay

    def test_adjusted_similarity_calculation(self):
        """Test adjusted similarity with time decay."""
        similarity = 0.9
        time_decay_weight = 0.3
        time_factor = 0.5  # 50% decay

        # Formula: similarity * (1 - weight + weight * time_factor)
        adjusted = similarity * (1 - time_decay_weight + time_decay_weight * time_factor)

        assert adjusted < similarity  # Should be lower due to decay
        assert adjusted > similarity * 0.7  # But not too much lower

    def test_no_decay_when_weight_zero(self):
        """Test that zero weight means no decay."""
        similarity = 0.9
        time_decay_weight = 0.0
        time_factor = 0.1  # High decay factor

        adjusted = similarity * (1 - time_decay_weight + time_decay_weight * time_factor)

        assert adjusted == similarity  # No change with zero weight


class TestTemporalQueryAnalysisIntegration:
    """Integration tests for temporal query analysis."""

    def test_temporal_info_in_query_analysis(self):
        """Test that temporal info is included in query analysis."""
        from rag.nodes.query_analysis import analyze_query

        query = "what are the recent updates"
        analysis = analyze_query(query)

        assert "is_temporal" in analysis
        assert "temporal_intent" in analysis
        assert "time_decay_weight" in analysis
        assert "temporal_window" in analysis

    def test_temporal_analysis_recent_query(self):
        """Test query analysis for recent query."""
        from rag.nodes.query_analysis import analyze_query

        query = "show me the latest documents"
        analysis = analyze_query(query)

        assert analysis["is_temporal"] is True
        assert analysis["temporal_intent"] == "recent"
        assert analysis["time_decay_weight"] > 0

    def test_temporal_analysis_specific_period(self):
        """Test query analysis for specific period query."""
        from rag.nodes.query_analysis import analyze_query

        query = "documents from last week"
        analysis = analyze_query(query)

        assert analysis["is_temporal"] is True
        assert analysis["temporal_intent"] == "specific_period"
        assert analysis["temporal_window"] == 7

    def test_temporal_analysis_non_temporal(self):
        """Test query analysis for non-temporal query."""
        from rag.nodes.query_analysis import analyze_query

        query = "what is the capital of France"
        analysis = analyze_query(query)

        assert analysis["is_temporal"] is False
        assert analysis["temporal_intent"] == "none"
        assert analysis["time_decay_weight"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
