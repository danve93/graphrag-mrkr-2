"""
HTML heading-aware chunker for structured documentation pages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from bs4 import BeautifulSoup, Tag

from core.token_counter import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class HtmlSection:
    heading_path: str
    section_title: Optional[str]
    anchor: Optional[str]
    blocks: List[str]


class HtmlHeadingChunker:
    """Chunk HTML content by heading hierarchy with token-aware sizing."""

    DROP_TAGS = {
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "form",
        "button",
        "svg",
        "canvas",
        "noscript",
    }
    DROP_SELECTORS = [
        ".toc",
        ".table-of-contents",
        ".breadcrumb",
        ".breadcrumbs",
        ".nav",
        ".sidebar",
        ".related",
        ".article-meta",
        ".article__meta",
        ".article-info",
    ]
    CONTENT_SELECTORS = [
        "article",
        "main",
        "div.article-body",
        "div.article-content",
        "div.article__body",
        "div#article-body",
        "div#article-content",
        "div.wiki-content",
        "div#main-content",
        "div#content",
        "div#content-body",
        "div.markdown-body",
    ]
    HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    BLOCK_TAGS = {"p", "li", "pre", "code", "table", "blockquote", "dt", "dd"}

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

    def extract_plain_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        self._strip_noise(soup)
        return self._clean_text(soup.get_text(" ", strip=True))

    def chunk_html(self, html: str) -> List[Dict[str, Any]]:
        soup = BeautifulSoup(html, "lxml")
        self._strip_noise(soup)
        document_url = self._extract_document_url(soup)
        document_title = self._extract_document_title(soup)
        container = self._select_main_container(soup)

        sections = self._extract_sections(container)
        chunks: List[Dict[str, Any]] = []
        for section in sections:
            section_text = "\n\n".join(section.blocks).strip()
            if not section_text and not section.heading_path:
                continue

            if self.include_heading_path and section.heading_path:
                section_text = f"{section.heading_path}\n\n{section_text}".strip()

            for chunk_text in self._split_text(section_text):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                chunk_meta = {
                    "section_title": section.section_title,
                    "heading_path": section.heading_path,
                    "section_anchor": section.anchor,
                    "document_url": document_url,
                    "document_title": document_title,
                    "token_count": self.token_counter.count(chunk_text),
                }
                chunks.append({"text": chunk_text, "metadata": chunk_meta})

        return chunks

    def _strip_noise(self, soup: BeautifulSoup) -> None:
        for tag in list(soup.find_all(self.DROP_TAGS)):
            tag.decompose()
        for selector in self.DROP_SELECTORS:
            for tag in list(soup.select(selector)):
                tag.decompose()

    def _select_main_container(self, soup: BeautifulSoup) -> Tag:
        best: Optional[Tag] = None
        best_len = 0
        for selector in self.CONTENT_SELECTORS:
            for candidate in soup.select(selector):
                text_len = len(candidate.get_text(" ", strip=True))
                if text_len > best_len:
                    best = candidate
                    best_len = text_len
        if best is not None and best_len > 0:
            return best
        return soup.body if soup.body is not None else soup

    def _extract_sections(self, container: Tag) -> List[HtmlSection]:
        sections: List[HtmlSection] = []
        heading_stack: List[Dict[str, Any]] = []
        current = HtmlSection("", None, None, [])

        for element in self._iter_blocks(container):
            if element.name in self.HEADING_TAGS:
                heading_text = self._clean_text(element.get_text(" ", strip=True))
                if not heading_text:
                    continue
                if current.blocks or current.heading_path:
                    sections.append(current)
                level = int(element.name[1])
                while heading_stack and heading_stack[-1]["level"] >= level:
                    heading_stack.pop()
                anchor = self._extract_anchor(element)
                heading_stack.append({"level": level, "text": heading_text, "anchor": anchor})
                heading_path = " > ".join(h["text"] for h in heading_stack)
                current = HtmlSection(heading_path, heading_text, anchor, [])
                continue

            block_text = self._extract_block_text(element)
            if block_text:
                current.blocks.append(block_text)

        if current.blocks or current.heading_path:
            sections.append(current)

        return sections

    def _iter_blocks(self, container: Tag) -> Iterable[Tag]:
        for element in container.find_all(
            list(self.HEADING_TAGS | self.BLOCK_TAGS), recursive=True
        ):
            if self._has_block_ancestor(element, container):
                continue
            yield element

    def _has_block_ancestor(self, element: Tag, container: Tag) -> bool:
        parent = element.parent
        while parent and parent != container:
            if parent.name in self.BLOCK_TAGS:
                return True
            parent = parent.parent
        return False

    def _extract_block_text(self, element: Tag) -> str:
        if element.name == "li":
            return f"- {self._clean_text(element.get_text(' ', strip=True))}"
        if element.name == "pre":
            return f"```\n{element.get_text()}\n```".strip()
        if element.name == "code":
            return f"`{element.get_text(strip=True)}`"
        if element.name == "table":
            return self._table_to_text(element)
        if element.name == "blockquote":
            text = self._clean_text(element.get_text(" ", strip=True))
            return f"> {text}" if text else ""
        return self._clean_text(element.get_text(" ", strip=True))

    def _table_to_text(self, table: Tag) -> str:
        rows = []
        for row in table.find_all("tr"):
            cells = [self._clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _extract_anchor(self, element: Tag) -> Optional[str]:
        anchor = element.get("id") or element.get("name")
        if anchor:
            return str(anchor)
        link = element.find("a", attrs={"id": True})
        if link:
            return str(link.get("id"))
        return None

    def _extract_document_url(self, soup: BeautifulSoup) -> Optional[str]:
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            return str(canonical.get("href"))
        og_url = soup.find("meta", attrs={"property": "og:url"})
        if og_url and og_url.get("content"):
            return str(og_url.get("content"))
        twitter_url = soup.find("meta", attrs={"name": "twitter:url"})
        if twitter_url and twitter_url.get("content"):
            return str(twitter_url.get("content"))
        return None

    def _extract_document_title(self, soup: BeautifulSoup) -> Optional[str]:
        if soup.title and soup.title.get_text(strip=True):
            return self._clean_text(soup.title.get_text(strip=True))
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return self._clean_text(h1.get_text(" ", strip=True))
        return None

    def _split_text(self, text: str) -> List[str]:
        units = [u.strip() for u in re.split(r"\n\s*\n", text) if u.strip()]
        expanded_units: List[str] = []
        for unit in units:
            unit_tokens = self.token_counter.count(unit)
            if unit_tokens > self.max_tokens:
                expanded_units.extend(self._split_long_unit(unit))
            else:
                expanded_units.append(unit)

        chunks: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0

        for unit in expanded_units:
            unit_tokens = self.token_counter.count(unit)
            if current_parts and current_tokens + unit_tokens > self.max_tokens:
                chunks.append("\n\n".join(current_parts))
                overlap = self.token_counter.tail_text(chunks[-1], self.overlap_tokens)
                current_parts = [overlap] if overlap else []
                current_tokens = self.token_counter.count(overlap)

            if current_parts and current_tokens + unit_tokens > self.target_tokens and current_tokens >= self.min_tokens:
                chunks.append("\n\n".join(current_parts))
                overlap = self.token_counter.tail_text(chunks[-1], self.overlap_tokens)
                current_parts = [overlap] if overlap else []
                current_tokens = self.token_counter.count(overlap)

            current_parts.append(unit)
            current_tokens += unit_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return self._merge_small_chunks(chunks)

    def _split_long_unit(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return self._split_by_tokens(text)

        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            tokens = self.token_counter.count(sentence)
            if current_tokens + tokens > self.max_tokens and current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sentence)
            current_tokens += tokens

        if current:
            chunks.append(" ".join(current))

        oversized = [c for c in chunks if self.token_counter.count(c) > self.max_tokens]
        if oversized:
            final: List[str] = []
            for chunk in chunks:
                if self.token_counter.count(chunk) > self.max_tokens:
                    final.extend(self._split_by_tokens(chunk))
                else:
                    final.append(chunk)
            return final
        return chunks

    def _split_by_tokens(self, text: str) -> List[str]:
        tokens = self.token_counter.encode(text)
        if not tokens:
            step = max(1, self.max_tokens * 4)
            return [text[i : i + step] for i in range(0, len(text), step)]

        step = max(1, self.max_tokens - self.overlap_tokens)
        chunks = []
        for idx in range(0, len(tokens), step):
            chunk_tokens = tokens[idx : idx + self.max_tokens]
            chunks.append(self.token_counter.decode(chunk_tokens))
        return [c for c in chunks if c.strip()]

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        merged: List[str] = []
        for chunk in chunks:
            chunk_tokens = self.token_counter.count(chunk)
            if merged:
                prev_tokens = self.token_counter.count(merged[-1])
                if chunk_tokens < self.min_tokens and prev_tokens + chunk_tokens <= self.max_tokens:
                    merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
                    continue
            merged.append(chunk)
        return merged

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
