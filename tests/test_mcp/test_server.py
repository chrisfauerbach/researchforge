"""Tests for MCP server tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from researchforge.mcp_server.server import (
    _mcp_jobs,
    get_briefing,
    get_status,
    ingest_document,
    list_briefings,
    mcp,
    query_corpus,
    research,
)


class TestToolDefinitions:
    def test_mcp_instance_exists(self):
        assert mcp is not None
        assert mcp.name == "researchforge"

    async def test_tools_are_registered(self):
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        expected = {
            "research", "query_corpus", "ingest_document",
            "list_briefings", "get_briefing", "get_status",
        }
        assert expected.issubset(tool_names)


class TestQueryCorpus:
    @patch("researchforge.rag.retriever.retrieve", new_callable=AsyncMock)
    @patch("researchforge.rag.store.VectorStore")
    async def test_returns_results(self, mock_store_cls, mock_retrieve):
        mock_retrieve.return_value = [
            {
                "text": "Sample chunk text",
                "source_path": "/docs/test.md",
                "source_type": "markdown",
                "content_type": "source",
                "section_h1": "Intro",
            }
        ]
        result = await query_corpus("test query", limit=5)
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["text"] == "Sample chunk text"
        assert data[0]["source_path"] == "/docs/test.md"


class TestIngestDocument:
    @patch("researchforge.mcp_server.server._get_repo")
    @patch("researchforge.rag.ingest.ingest_file", new_callable=AsyncMock)
    @patch("researchforge.rag.store.VectorStore")
    async def test_processes_file(self, mock_store_cls, mock_ingest, mock_get_repo):
        mock_repo = AsyncMock()
        mock_get_repo.return_value = mock_repo

        mock_result = MagicMock()
        mock_result.document_id = "doc-123"
        mock_result.chunk_count = 5
        mock_result.source_type = "markdown"
        mock_result.skipped = False
        mock_result.reason = ""
        mock_ingest.return_value = mock_result

        result = await ingest_document("/tmp/test.md")
        data = json.loads(result)
        assert data["document_id"] == "doc-123"
        assert data["chunk_count"] == 5
        assert data["skipped"] is False


class TestListBriefings:
    @patch("researchforge.mcp_server.server._get_repo")
    async def test_returns_briefings(self, mock_get_repo):
        mock_repo = AsyncMock()
        mock_repo.list_briefings.return_value = [
            {
                "briefing_id": "b1",
                "research_question": "What is RAG?",
                "status": "completed",
                "started_at": "2024-01-01T00:00:00",
                "quality_score": 0.8,
            }
        ]
        mock_get_repo.return_value = mock_repo

        result = await list_briefings(limit=10, status="all")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["briefing_id"] == "b1"
        assert data[0]["research_question"] == "What is RAG?"


class TestGetBriefing:
    @patch("researchforge.mcp_server.server._get_repo")
    async def test_returns_briefing(self, mock_get_repo):
        mock_repo = AsyncMock()
        mock_repo.get_briefing.return_value = {
            "briefing_id": "b1",
            "research_question": "What is RAG?",
            "status": "completed",
            "briefing_markdown": "# RAG Overview\n\nContent here.",
            "quality_score": 0.8,
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:05:00",
        }
        mock_get_repo.return_value = mock_repo

        result = await get_briefing("b1")
        data = json.loads(result)
        assert data["briefing_id"] == "b1"
        assert "RAG Overview" in data["briefing_markdown"]

    @patch("researchforge.mcp_server.server._get_repo")
    async def test_not_found(self, mock_get_repo):
        mock_repo = AsyncMock()
        mock_repo.get_briefing.return_value = None
        mock_get_repo.return_value = mock_repo

        result = await get_briefing("nonexistent")
        data = json.loads(result)
        assert "error" in data


class TestGetStatus:
    async def test_returns_job_info(self):
        _mcp_jobs["test-job"] = {
            "status": "running",
            "question": "Test question",
            "result": None,
        }
        try:
            result = await get_status("test-job")
            data = json.loads(result)
            assert data["job_id"] == "test-job"
            assert data["status"] == "running"
            assert data["question"] == "Test question"
        finally:
            _mcp_jobs.pop("test-job", None)

    async def test_job_not_found(self):
        result = await get_status("nonexistent-job")
        data = json.loads(result)
        assert "error" in data


class TestResearch:
    @patch("researchforge.mcp_server.server.asyncio.create_task")
    async def test_creates_job(self, mock_create_task):
        mock_create_task.return_value = MagicMock()
        result = await research("What is quantum computing?", depth="quick")
        data = json.loads(result)
        assert "job_id" in data
        assert data["status"] == "queued"
        job_id = data["job_id"]
        assert job_id in _mcp_jobs
        assert _mcp_jobs[job_id]["question"] == "What is quantum computing?"
        # Clean up
        _mcp_jobs.pop(job_id, None)
