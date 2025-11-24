"""Community summarization utilities leveraging entity and TextUnit metadata."""

import logging
from typing import Any, Dict, List, Optional

from core.graph_db import graph_db
from core.llm import llm_manager

logger = logging.getLogger(__name__)


class CommunitySummarizer:
    """Generate and persist summaries for entity communities."""

    def __init__(self, text_unit_limit: int = 5, excerpt_length: int = 500):
        """Configure default limits for exemplar selection."""

        self.text_unit_limit = text_unit_limit
        self.excerpt_length = excerpt_length

    def summarize_levels(self, levels: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Walk communities from lowest to higher levels and summarize each one."""

        levels_to_process = levels or graph_db.get_community_levels()
        summaries: List[Dict[str, Any]] = []

        for level in sorted(levels_to_process):
            communities = graph_db.get_communities_for_level(level)
            logger.info(
                "Summarizing %s communities at level %s", len(communities), level
            )

            for community in communities:
                community_id = community.get("community_id")
                entities = community.get("entities", [])

                if community_id is None:
                    logger.warning(
                        "Skipping community without identifier at level %s", level
                    )
                    continue

                text_units = graph_db.get_text_units_for_entities(
                    [entity.get("id") for entity in entities if entity.get("id")],
                    limit=self.text_unit_limit,
                )

                text_unit_payloads = self._build_text_unit_payloads(text_units)

                summary_text = self._generate_summary(
                    community_id=community_id,
                    level=level,
                    entities=entities,
                    text_units=text_units,
                )

                graph_db.upsert_community_summary(
                    community_id=community_id,
                    level=level,
                    summary=summary_text,
                    member_entities=entities,
                    exemplar_text_units=text_unit_payloads,
                )

                summaries.append(
                    {
                        "community_id": community_id,
                        "level": level,
                        "summary": summary_text,
                        "member_entities": entities,
                        "exemplar_text_units": text_unit_payloads,
                    }
                )

        return summaries

    def _generate_summary(
        self,
        community_id: int,
        level: int,
        entities: List[Dict[str, Any]],
        text_units: List[Dict[str, Any]],
    ) -> str:
        """Invoke the configured LLM to summarize a single community."""

        if not entities:
            logger.warning(
                "No entities found for community %s at level %s; skipping summary",
                community_id,
                level,
            )
            return ""

        entity_lines = []
        for entity in entities:
            entity_lines.append(
                f"- {entity.get('name', '')} ({entity.get('type', '')}) "
                f"[importance: {entity.get('importance_score', 0.0)}]: "
                f"{entity.get('description', '')}"
            )

        text_unit_lines = []
        for unit in text_units:
            excerpt = self._trim_excerpt(unit.get("content", ""))
            text_unit_lines.append(
                f"- TextUnit {unit.get('id', '')} (doc {unit.get('document_id', 'unknown')}): {excerpt}"
            )

        prompt = (
            f"You are summarizing community {community_id} at level {level}.\n"
            f"Member entities:\n{chr(10).join(entity_lines)}\n\n"
            f"Representative TextUnits:\n{chr(10).join(text_unit_lines)}\n\n"
            "Provide a concise overview (2-4 sentences) capturing the main theme and how the entities relate. "
            "Highlight notable member entities and what the exemplar TextUnits reveal about this community."
        )

        system_message = (
            "You are a graph intelligence assistant. Summaries must stay faithful to the provided "
            "entities and TextUnits. Keep provenance by mentioning entity names and referencing the "
            "TextUnit identifiers when possible."
        )

        return llm_manager.generate_response(
            prompt=prompt,
            system_message=system_message,
            temperature=0.2,
            max_tokens=500,
        )

    def _build_text_unit_payloads(self, text_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize TextUnit metadata for persistence and downstream use."""

        payloads: List[Dict[str, Any]] = []
        for unit in text_units:
            payloads.append(
                {
                    "id": unit.get("id"),
                    "document_id": unit.get("document_id"),
                    "metadata": unit.get("metadata", {}),
                    "excerpt": self._trim_excerpt(unit.get("content", "")),
                }
            )
        return payloads

    def _trim_excerpt(self, content: str) -> str:
        """Limit exemplar snippets to the configured maximum length."""

        if len(content) <= self.excerpt_length:
            return content
        return content[: self.excerpt_length].rstrip() + "…"


# Shared instance for use across the application
community_summarizer = CommunitySummarizer()
