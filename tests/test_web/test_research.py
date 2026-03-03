"""Tests for the research routes."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from researchforge.web.app import get_repo
from researchforge.web.routes.research import _active_pipelines


class TestDashboard:
    async def test_dashboard_renders(self, web_client):
        resp = await web_client.get("/")
        assert resp.status_code == 200
        assert "Research" in resp.text
        assert "question" in resp.text


class TestStartResearch:
    async def test_start_research_returns_card(self, web_client):
        _active_pipelines.clear()
        with patch(
            "researchforge.web.routes.research"
            "._run_pipeline_with_events",
            new_callable=AsyncMock,
        ):
            resp = await web_client.post(
                "/api/research",
                data={"question": "What is RAG?", "depth": "standard"},
            )
            assert resp.status_code == 200
            assert "What is RAG?" in resp.text
        _active_pipelines.clear()

    async def test_start_research_tracks_pipeline(self, web_client):
        _active_pipelines.clear()
        with patch(
            "researchforge.web.routes.research"
            "._run_pipeline_with_events",
            new_callable=AsyncMock,
        ):
            await web_client.post(
                "/api/research",
                data={"question": "Test tracking", "depth": "quick"},
            )
            assert len(_active_pipelines) == 1
            info = next(iter(_active_pipelines.values()))
            assert info["question"] == "Test tracking"
            assert info["depth"] == "quick"
            assert info["status"] == "running"
        _active_pipelines.clear()


class TestListPipelines:
    async def test_empty_pipelines(self, web_client):
        _active_pipelines.clear()
        resp = await web_client.get("/api/pipelines")
        assert resp.status_code == 200
        assert "No pipelines yet" in resp.text

    async def test_pipelines_shows_active(self, web_client):
        _active_pipelines.clear()
        _active_pipelines["test-job"] = {
            "question": "Active question?",
            "status": "running",
            "depth": "standard",
            "completed_stages": ["planner"],
            "current_stage": "gatherer",
            "error_stages": [],
        }
        resp = await web_client.get("/api/pipelines")
        assert resp.status_code == 200
        assert "Active question?" in resp.text
        _active_pipelines.clear()

    async def test_pipelines_shows_completed_from_db(self, web_client):
        _active_pipelines.clear()
        repo = await get_repo()
        await repo.insert_briefing(
            "db-job", "DB question?", status="completed"
        )
        resp = await web_client.get("/api/pipelines")
        assert resp.status_code == 200
        assert "DB question?" in resp.text
        _active_pipelines.clear()


class TestSSEStream:
    async def test_stream_receives_events(self, web_client):
        from researchforge.web.events import get_event_bus

        event_bus = get_event_bus()
        job_id = "sse-test"

        # Publish events after a short delay
        async def publish_events():
            await asyncio.sleep(0.1)
            await event_bus.publish_stage_start(
                job_id, "planner", "test-model"
            )
            await event_bus.publish_complete(
                job_id, briefing_id=job_id
            )

        task = asyncio.create_task(publish_events())

        # Read the SSE stream
        buffer = ""
        async with web_client.stream(
            "GET", f"/api/pipelines/{job_id}/stream"
        ) as resp:
            assert resp.status_code == 200
            async for chunk in resp.aiter_text():
                buffer += chunk
                if "pipeline_complete" in buffer:
                    break

        await task
        assert "stage_start" in buffer
        assert "pipeline_complete" in buffer
