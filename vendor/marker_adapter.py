"""Adapter to use Marker conversion when available.

This module wraps optional imports of `marker-pdf` and performs PDF conversion
returning a simple dict with `content` and `metadata`. Returns None when Marker
is unavailable or conversion fails.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def convert_pdf(file_path: Path, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a PDF using Marker if installed.

    Args:
        file_path: Path to PDF file
        config: Marker configuration dict

    Returns:
        Dict with keys `content` and `metadata`, or None on failure/unavailable.
    """
    try:
        from marker.converters.pdf import PdfConverter  # type: ignore
        from marker.models import create_model_dict  # type: ignore
        from marker.output import text_from_rendered  # type: ignore
    except Exception as e:
        logger.info("Marker library not available: %s", e)
        return None

    try:
        artifact_dict = create_model_dict()
        llm_service_arg = config.get("llm_service")
        
        # If use_llm is enabled and no explicit llm_service provided, prefer passing
        # the dotted class path string to Marker (it will instantiate internally).
        # Also keep llm keys in the converter config so Marker can use them.
        if config.get("use_llm") and not llm_service_arg:
            try:
                # Prefer the dotted path that Marker expects rather than instantiating
                # the service here. Marker internals may call `.rsplit` on this value.
                llm_service_arg = "marker.services.openai.OpenAIService"
                logger.info(f"Marker configured to use OpenAI LLM service (class path), model: {config.get('llm_model', 'default')}")
            except Exception as e:
                logger.warning(f"Failed to configure Marker OpenAIService class path: {e}, LLM features disabled")
                llm_service_arg = None
        
        # Build converter; PdfConverter accepts `config` and optional `llm_service`
        # Include LLM config keys in the converter config so Marker can instantiate
        # the service itself when given a dotted class path.
        converter_config = {k: v for k, v in config.items() if k not in ["llm_service"]}
        converter = PdfConverter(
            artifact_dict=artifact_dict,
            config=converter_config,
            llm_service=llm_service_arg if llm_service_arg else None,
        )
        rendered = converter(str(file_path))
        text, ext, images = text_from_rendered(rendered)
        if not text:
            logger.warning("Marker produced empty output for %s", file_path)
            return None

        marker_meta: Dict[str, Any] = {}
        try:
            marker_meta = getattr(rendered, "metadata", {}) or {}
        except Exception:
            marker_meta = {}

        return {
            "content": text,
            "metadata": {
                "conversion_pipeline": "marker",
                "marker_output_ext": ext,
                "marker_page_count": getattr(converter, "page_count", None),
                "marker_metadata": marker_meta,
            },
        }
    except Exception as e:
        logger.error("Marker conversion failed for %s: %s", file_path, e)
        return None
