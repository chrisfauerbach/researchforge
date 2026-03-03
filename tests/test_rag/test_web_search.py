"""Tests for the web search module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from researchforge.config import WebSearchConfig
from researchforge.rag.web_search import (
    _extract_text,
    _to_chunks,
    web_search_for_question,
)

_P = "researchforge.rag.web_search"


class TestExtractText:
    def test_strips_scripts(self):
        html = "<html><body><script>alert(1)</script><p>Hello</p></body></html>"
        assert "alert" not in _extract_text(html, 8000)
        assert "Hello" in _extract_text(html, 8000)

    def test_strips_aside(self):
        html = "<html><body><aside>Sidebar</aside><p>Main content</p></body></html>"
        result = _extract_text(html, 8000)
        assert "Sidebar" not in result
        assert "Main content" in result

    def test_respects_max_chars(self):
        html = "<html><body><p>" + "x" * 500 + "</p></body></html>"
        result = _extract_text(html, 100)
        assert len(result) <= 100

    def test_handles_empty_html(self):
        result = _extract_text("", 8000)
        assert result == ""


class TestToChunks:
    def test_produces_correct_fields(self):
        chunks = _to_chunks("https://example.com", "Example Title", "Some text content here.")
        assert len(chunks) >= 1
        c = chunks[0]
        assert c["source_path"] == "https://example.com"
        assert c["section_h1"] == "Example Title"
        assert c["source_type"] == "web"
        assert c["content_type"] == "web_search"
        assert c["text"]

    def test_deterministic_ids(self):
        chunks_a = _to_chunks("https://example.com", "Title", "Some content")
        chunks_b = _to_chunks("https://example.com", "Title", "Some content")
        assert chunks_a[0]["chunk_id"] == chunks_b[0]["chunk_id"]
        assert chunks_a[0]["chunk_id"].startswith("web:")


class TestWebSearchForQuestion:
    async def test_returns_chunks_on_success(self):
        cfg = WebSearchConfig(max_results=2, max_page_chars=8000, fetch_timeout_seconds=5)
        mock_hits = [
            {"url": "https://a.com", "title": "Page A"},
            {"url": "https://b.com", "title": "Page B"},
        ]
        mock_html = "<html><body><p>Relevant info about the topic.</p></body></html>"

        with (
            patch(f"{_P}._ddg_search", new_callable=AsyncMock) as mock_ddg,
            patch(f"{_P}._fetch_page", new_callable=AsyncMock) as mock_fetch,
        ):
            mock_ddg.return_value = mock_hits
            mock_fetch.return_value = mock_html

            chunks = await web_search_for_question("test query", cfg)

        assert len(chunks) >= 2  # At least one chunk per page
        assert all(c["source_type"] == "web" for c in chunks)

    async def test_returns_empty_on_ddg_failure(self):
        cfg = WebSearchConfig(max_results=2)

        with patch(f"{_P}._ddg_search", new_callable=AsyncMock) as mock_ddg:
            mock_ddg.side_effect = Exception("rate limited")

            chunks = await web_search_for_question("test query", cfg)

        assert chunks == []

    async def test_skips_failed_fetches(self):
        cfg = WebSearchConfig(max_results=2, max_page_chars=8000, fetch_timeout_seconds=5)
        mock_hits = [
            {"url": "https://good.com", "title": "Good"},
            {"url": "https://bad.com", "title": "Bad"},
        ]

        with (
            patch(f"{_P}._ddg_search", new_callable=AsyncMock) as mock_ddg,
            patch(f"{_P}._fetch_page", new_callable=AsyncMock) as mock_fetch,
        ):
            mock_ddg.return_value = mock_hits
            # First page succeeds, second fails
            mock_fetch.side_effect = [
                "<html><body><p>Good content here.</p></body></html>",
                None,
            ]

            chunks = await web_search_for_question("test query", cfg)

        # Only chunks from the successful page
        assert len(chunks) >= 1
        assert all(c["source_path"] == "https://good.com" for c in chunks)
