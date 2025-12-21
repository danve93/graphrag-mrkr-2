"""
Docling Hybrid chunker adapter.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional

from core.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class DoclingTokenizerAdapter:
    """Adapter for docling chunkers that expect a tokenizer object."""

    def __init__(self, counter: TokenCounter) -> None:
        self.counter = counter

    def __call__(self, text: str) -> List[int]:
        return self.counter.encode(text)

    def encode(self, text: str) -> List[int]:
        return self.counter.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.counter.decode(tokens)


class DoclingHybridChunker:
    """Wrap Docling's HybridChunker with safe fallbacks."""

    def __init__(
        self,
        target_tokens: int,
        min_tokens: int,
        max_tokens: int,
        overlap_tokens: int,
        tokenizer_name: str = "cl100k_base",
        include_heading_path: bool = True,
    ) -> None:
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.include_heading_path = include_heading_path
        self.token_counter = TokenCounter(tokenizer_name)
        self._chunker = self._build_chunker()

    def is_available(self) -> bool:
        return self._chunker is not None

    def chunk_document(self, docling_document: Any) -> List[Dict[str, Any]]:
        if not self._chunker:
            return []
        chunks = self._iterate_chunks(docling_document)
        output: List[Dict[str, Any]] = []
        for chunk in chunks:
            text = self._extract_text(chunk)
            if not text:
                continue
            metadata = self._extract_metadata(chunk)
            
            # Prepend heading path if enabled and available
            if self.include_heading_path and metadata.get("heading_path"):
                heading_path = metadata["heading_path"]
                text = f"{heading_path}\n\n{text}".strip()

            metadata["token_count"] = self.token_counter.count(text)
            output.append({"text": text, "metadata": metadata})
        return output

    def _build_chunker(self):
        try:
            from docling.chunking import HybridChunker  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.info("Docling HybridChunker unavailable: %s", exc)
            return None

        kwargs: Dict[str, Any] = {}
        try:
            sig = inspect.signature(HybridChunker.__init__)
        except Exception:  # pragma: no cover
            sig = None

        if sig:
            params = sig.parameters
            if "tokenizer" in params:
                kwargs["tokenizer"] = DoclingTokenizerAdapter(self.token_counter)
            if "target_tokens" in params:
                kwargs["target_tokens"] = self.target_tokens
            if "chunk_size" in params:
                kwargs["chunk_size"] = self.target_tokens
            if "min_tokens" in params:
                kwargs["min_tokens"] = self.min_tokens
            if "max_tokens" in params:
                kwargs["max_tokens"] = self.max_tokens
            if "overlap_tokens" in params:
                kwargs["overlap_tokens"] = self.overlap_tokens
            if "overlap" in params:
                kwargs["overlap"] = self.overlap_tokens

        try:
            return HybridChunker(**kwargs)
        except Exception as exc:  # pragma: no cover - version mismatch fallback
            logger.warning("Failed to init Docling HybridChunker: %s", exc)
            try:
                return HybridChunker()
            except Exception:
                return None

    def _iterate_chunks(self, docling_document: Any) -> List[Any]:
        if hasattr(self._chunker, "chunk"):
            return list(self._chunker.chunk(docling_document))
        if callable(self._chunker):
            return list(self._chunker(docling_document))
        return []

    def _extract_text(self, chunk: Any) -> str:
        for attr in ("text", "content"):
            value = getattr(chunk, attr, None)
            if isinstance(value, str):
                return value.strip()
        for method in ("get_text", "to_text"):
            fn = getattr(chunk, method, None)
            if callable(fn):
                try:
                    value = fn()
                except Exception:
                    continue
                if isinstance(value, str):
                    return value.strip()
        return ""

    def _extract_metadata(self, chunk: Any) -> Dict[str, Any]:
        meta = {}
        raw_meta = getattr(chunk, "metadata", None) or getattr(chunk, "meta", None)
        if isinstance(raw_meta, dict):
            heading_path = raw_meta.get("heading_path")
            headings = raw_meta.get("headings")
            section_title = raw_meta.get("section_title") or raw_meta.get("heading")
            page = raw_meta.get("page") or raw_meta.get("page_number")
            if not heading_path and isinstance(headings, list):
                heading_path = " > ".join(str(h) for h in headings if h)
            if heading_path:
                meta["heading_path"] = heading_path
            if section_title:
                meta["section_title"] = section_title
            if page is not None:
                meta["page"] = page

        page_attr = getattr(chunk, "page", None)
        if page_attr is not None and "page" not in meta:
            meta["page"] = page_attr

        return meta
