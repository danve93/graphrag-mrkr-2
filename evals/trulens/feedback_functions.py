"""
Custom TruLens feedback functions for Amber GraphRAG evaluation.

Defines feedback evaluators for:
- Answer relevance (built-in)
- Context relevance (built-in)
- Groundedness (built-in)
- Graph reasoning quality (custom)
- Latency check (custom)

Updated for TruLens v2.5+ API.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AmberFeedbackFunctions:
    """
    TruLens feedback functions for Amber GraphRAG pipeline.

    Feedback functions evaluate:
    - Answer relevance: Does answer address the query?
    - Context relevance: Are retrieved chunks relevant to the query?
    - Groundedness: Is answer faithful to retrieved context?
    - Graph reasoning quality: Did graph enrichment add value?
    - Latency check: Is response time acceptable?
    """

    def __init__(self, llm_provider: str = "openai", config: Optional[Dict] = None):
        """
        Initialize feedback functions.

        Args:
            llm_provider: LLM provider for feedback evaluation
                         (reuses existing llm_manager config)
            config: Optional configuration dict with thresholds
        """
        self.llm_provider = llm_provider
        self.config = config or {}

        # Initialize TruLens feedback providers
        try:
            from trulens.providers.openai import OpenAI as fOpenAI

            # TruLens OpenAI feedback provider (new v2.5+ API)
            self.provider = fOpenAI()

            logger.info(f"Feedback functions initialized (provider={llm_provider})")

        except ImportError as e:
            logger.error(
                f"Failed to import TruLens feedback modules: {e}. "
                "Install with: pip install trulens trulens-providers-openai"
            )
            raise

    def get_feedback_functions(self):
        """
        Get list of TruLens feedback functions for pipeline evaluation.

        Returns:
            List of Feedback objects for TruLens recording
        """
        from trulens.core import Feedback

        feedbacks = []

        try:
            # 1. Answer Relevance: Does the answer address the question?
            f_answer_relevance = (
                Feedback(
                    self.provider.relevance_with_cot_reasons,
                    name="Answer Relevance",
                )
                .on_input()
                .on_output()
            )
            feedbacks.append(f_answer_relevance)

            # 2. Groundedness: Is answer faithful to retrieved context?
            f_groundedness = (
                Feedback(
                    self.provider.groundedness_measure_with_cot_reasons,
                    name="Groundedness",
                )
                .on_context(collect_list=True)
                .on_output()
            )
            feedbacks.append(f_groundedness)

            # 3. Context Relevance: Are retrieved chunks relevant to query?
            f_context_relevance = (
                Feedback(
                    self.provider.context_relevance_with_cot_reasons,
                    name="Context Relevance",
                )
                .on_input()
                .on_context(collect_list=False)
                .aggregate(np.mean)
            )
            feedbacks.append(f_context_relevance)

            logger.info(f"Registered {len(feedbacks)} feedback functions")

        except Exception as e:
            logger.error(f"Error creating feedback functions: {e}", exc_info=True)

        return feedbacks

    def _graph_reasoning_quality(self, result: Dict[str, Any]) -> float:
        """
        Custom feedback: Evaluate graph reasoning quality.

        Checks if graph enrichment (entities, relationships, paths) added
        value beyond basic vector retrieval.

        Args:
            result: GraphRAG query result dict

        Returns:
            float: Quality score 0.0-1.0
        """
        try:
            # Check for graph-specific signals in sources
            sources = result.get("sources", [])
            if not sources:
                return 0.0

            # Count sources with graph enrichment
            graph_enhanced_count = 0
            for source in sources:
                metadata = source.get("metadata", {})

                # Check for entity-based retrieval
                if metadata.get("entity_id") or metadata.get("entity_name"):
                    graph_enhanced_count += 1
                    continue

                # Check for relationship-based retrieval
                if metadata.get("relationship_type"):
                    graph_enhanced_count += 1
                    continue

                # Check for path-based retrieval
                if metadata.get("path_length") and metadata.get("path_length") > 1:
                    graph_enhanced_count += 1
                    continue

            # Calculate percentage of graph-enhanced sources
            if len(sources) == 0:
                return 0.0

            graph_ratio = graph_enhanced_count / len(sources)

            # Scoring:
            # - 1.0: >50% graph-enhanced (strong graph usage)
            # - 0.5-1.0: 25-50% graph-enhanced (moderate usage)
            # - 0.0-0.5: <25% graph-enhanced (minimal usage)
            if graph_ratio >= 0.5:
                return 1.0
            elif graph_ratio >= 0.25:
                return 0.5 + (graph_ratio - 0.25) * 2.0  # Scale 0.25-0.5 to 0.5-1.0
            else:
                return graph_ratio * 2.0  # Scale 0.0-0.25 to 0.0-0.5

        except Exception as e:
            logger.warning(f"Error in graph_reasoning_quality: {e}")
            return 0.5  # Neutral score on error

    def _latency_check(self, result: Dict[str, Any]) -> float:
        """
        Custom feedback: Evaluate if latency is acceptable.

        Checks if total query latency is below configured threshold.

        Args:
            result: GraphRAG query result dict

        Returns:
            float: 1.0 if acceptable, <1.0 if slow, 0.0 if timeout
        """
        try:
            # Get total latency from stages
            stages = result.get("stages", [])
            total_ms = sum(stage.get("duration_ms", 0) for stage in stages)

            # Get threshold from config (default 5000ms = 5 seconds)
            threshold_ms = self.config.get("latency_max_ms", 5000)

            # Scoring:
            # - 1.0: Below threshold (good)
            # - 0.7-1.0: Within 1.5x threshold (acceptable)
            # - 0.0-0.7: Above 1.5x threshold (problematic)
            if total_ms <= threshold_ms:
                return 1.0
            elif total_ms <= threshold_ms * 1.5:
                # Linear decay from 1.0 to 0.7
                excess_ratio = (total_ms - threshold_ms) / (threshold_ms * 0.5)
                return 1.0 - (excess_ratio * 0.3)
            else:
                # Exponential decay for severe latency
                excess_ratio = total_ms / (threshold_ms * 1.5)
                return max(0.0, 0.7 / excess_ratio)

        except Exception as e:
            logger.warning(f"Error in latency_check: {e}")
            return 0.5  # Neutral score on error


def create_feedback_functions(config: Optional[Dict] = None):
    """
    Factory function to create AmberFeedbackFunctions instance.

    Args:
        config: Optional configuration dict

    Returns:
        AmberFeedbackFunctions instance
    """
    llm_provider = config.get("feedback", {}).get("provider", "openai") if config else "openai"

    feedback_funcs = AmberFeedbackFunctions(
        llm_provider=llm_provider,
        config=config.get("feedback", {}).get("thresholds", {}) if config else {}
    )

    return feedback_funcs
