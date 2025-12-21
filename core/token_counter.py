"""
Token counting utilities for chunking and overlap management.
"""

from __future__ import annotations

import logging
from typing import List, Optional

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None
    HAS_TIKTOKEN = False

logger = logging.getLogger(__name__)


class TokenCounter:
    """Count tokens with a fixed tokenizer and provide overlap extraction."""

    def __init__(self, tokenizer_name: str = "cl100k_base") -> None:
        self.tokenizer_name = tokenizer_name
        self._encoder = None
        if HAS_TIKTOKEN:
            try:
                self._encoder = tiktoken.get_encoding(tokenizer_name)  # type: ignore[arg-type]
            except Exception as exc:
                logger.warning(
                    "Failed to load tokenizer '%s': %s; falling back to cl100k_base",
                    tokenizer_name,
                    exc,
                )
                try:
                    self._encoder = tiktoken.get_encoding("cl100k_base")  # type: ignore[arg-type]
                except Exception:
                    self._encoder = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception as exc:
                logger.warning("Token counting failed, using approximation: %s", exc)
        return max(1, len(text) // 4)

    def encode(self, text: str) -> List[int]:
        if self._encoder is not None and text:
            try:
                return self._encoder.encode(text)
            except Exception:
                return []
        return []

    def decode(self, tokens: List[int]) -> str:
        if self._encoder is not None and tokens:
            try:
                return self._encoder.decode(tokens)
            except Exception:
                return ""
        return ""

    def tail_text(self, text: str, overlap_tokens: int) -> str:
        """Return the trailing text covering the last overlap_tokens tokens."""
        if overlap_tokens <= 0 or not text:
            return ""
        if self._encoder is not None:
            tokens = self.encode(text)
            if not tokens:
                return ""
            tail = tokens[-overlap_tokens:] if len(tokens) > overlap_tokens else tokens
            return self.decode(tail)
        # Approximate overlap with character slicing when tokenizer is unavailable.
        char_count = max(1, overlap_tokens * 4)
        return text[-char_count:]
