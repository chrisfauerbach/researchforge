"""Tests for the corpus routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


class TestCorpusPage:
    async def test_corpus_page_renders(self, web_client):
        resp = await web_client.get("/corpus")
        assert resp.status_code == 200
        assert "Corpus" in resp.text


class TestCorpusSearch:
    async def test_search_empty_query(self, web_client):
        resp = await web_client.get("/api/corpus/search?q=")
        assert resp.status_code == 200
        assert resp.text == ""

    async def test_search_short_query(self, web_client):
        resp = await web_client.get("/api/corpus/search?q=a")
        assert resp.status_code == 200
        assert resp.text == ""

    async def test_search_returns_results(self, web_client):
        mock_results = [
            {
                "text": "RAG is a technique...",
                "score": 0.95,
                "metadata": {"source_path": "doc.md"},
            }
        ]
        with patch(
            "researchforge.rag.retriever.retrieve",
            new_callable=AsyncMock,
            return_value=mock_results,
        ), patch(
            "researchforge.rag.store.VectorStore",
            return_value=MagicMock(),
        ):
            resp = await web_client.get(
                "/api/corpus/search?q=RAG technique"
            )
            assert resp.status_code == 200

    async def test_search_handles_error(self, web_client):
        with patch(
            "researchforge.rag.retriever.retrieve",
            new_callable=AsyncMock,
            side_effect=Exception("search failed"),
        ), patch(
            "researchforge.rag.store.VectorStore",
            return_value=MagicMock(),
        ):
            resp = await web_client.get(
                "/api/corpus/search?q=test query"
            )
            assert resp.status_code == 200
            # Should return empty results on error, not crash


class TestCorpusStats:
    async def test_stats_html(self, web_client):
        with patch(
            "researchforge.rag.store.VectorStore"
        ) as MockStore:
            MockStore.return_value.count.return_value = 42
            resp = await web_client.get("/api/corpus/stats")
            assert resp.status_code == 200
            assert "42" in resp.text

    async def test_stats_json(self, web_client):
        with patch(
            "researchforge.rag.store.VectorStore"
        ) as MockStore:
            MockStore.return_value.count.return_value = 10
            resp = await web_client.get("/api/corpus/stats/json")
            assert resp.status_code == 200
            data = resp.json()
            assert data["chunks"] == 10
            assert "documents" in data

    async def test_stats_handles_store_error(self, web_client):
        with patch(
            "researchforge.rag.store.VectorStore"
        ) as MockStore:
            MockStore.return_value.count.side_effect = Exception("no store")
            resp = await web_client.get("/api/corpus/stats/json")
            assert resp.status_code == 200
            data = resp.json()
            assert data["chunks"] == 0


class TestCorpusIngest:
    async def test_ingest_unsupported_file_type(self, web_client):
        resp = await web_client.post(
            "/api/corpus/ingest",
            files={
                "file": (
                    "test.exe",
                    b"binary content",
                    "application/octet-stream",
                )
            },
        )
        assert resp.status_code == 200
        assert "Unsupported file type" in resp.text

    async def test_ingest_success(self, web_client):
        mock_result = MagicMock()
        mock_result.skipped = False
        mock_result.chunk_count = 5
        with patch(
            "researchforge.rag.ingest.ingest_file",
            new_callable=AsyncMock,
            return_value=mock_result,
        ), patch(
            "researchforge.rag.store.VectorStore",
            return_value=MagicMock(),
        ):
            resp = await web_client.post(
                "/api/corpus/ingest",
                files={
                    "file": (
                        "test.md",
                        b"# Hello\nContent",
                        "text/markdown",
                    )
                },
            )
            assert resp.status_code == 200
            assert "Ingested test.md" in resp.text
            assert "5 chunks" in resp.text

    async def test_ingest_skipped(self, web_client):
        mock_result = MagicMock()
        mock_result.skipped = True
        with patch(
            "researchforge.rag.ingest.ingest_file",
            new_callable=AsyncMock,
            return_value=mock_result,
        ), patch(
            "researchforge.rag.store.VectorStore",
            return_value=MagicMock(),
        ):
            resp = await web_client.post(
                "/api/corpus/ingest",
                files={
                    "file": (
                        "dup.txt",
                        b"duplicate",
                        "text/plain",
                    )
                },
            )
            assert resp.status_code == 200
            assert "Skipped" in resp.text

    async def test_ingest_error(self, web_client):
        with patch(
            "researchforge.rag.ingest.ingest_file",
            new_callable=AsyncMock,
            side_effect=Exception("ingest boom"),
        ), patch(
            "researchforge.rag.store.VectorStore",
            return_value=MagicMock(),
        ):
            resp = await web_client.post(
                "/api/corpus/ingest",
                files={
                    "file": (
                        "err.pdf",
                        b"pdf bytes",
                        "application/pdf",
                    )
                },
            )
            assert resp.status_code == 200
            assert "Error" in resp.text
