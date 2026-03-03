"""Tests for the briefings routes."""

from __future__ import annotations

from researchforge.web.app import get_repo


async def _seed_briefing(briefing_id="b-1", question="Test?"):
    """Insert a briefing directly via the repository."""
    repo = await get_repo()
    await repo.insert_briefing(briefing_id, question, status="completed")
    await repo.update_briefing(
        briefing_id,
        status="completed",
        briefing_markdown="# Test Briefing\n\nSome content.",
        pipeline_trace=[{"agent": "planner", "duration_ms": 100}],
    )


class TestBriefingsPage:
    async def test_briefings_list_empty(self, web_client):
        resp = await web_client.get("/briefings")
        assert resp.status_code == 200
        assert "Briefings" in resp.text

    async def test_briefings_list_with_data(self, web_client):
        await _seed_briefing("b-list-1", "What is RAG?")
        resp = await web_client.get("/briefings")
        assert resp.status_code == 200
        assert "What is RAG?" in resp.text


class TestBriefingDetail:
    async def test_briefing_not_found(self, web_client):
        resp = await web_client.get("/briefings/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.text.lower()

    async def test_briefing_detail_renders_markdown(self, web_client):
        await _seed_briefing("b-detail-1", "Explain RAG")
        resp = await web_client.get("/briefings/b-detail-1")
        assert resp.status_code == 200
        # Markdown should be rendered to HTML (toc extension adds id attr)
        assert "Test Briefing</h1>" in resp.text
        assert "Some content." in resp.text


class TestBriefingsApi:
    async def test_list_briefings_json_empty(self, web_client):
        resp = await web_client.get("/api/briefings")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_briefings_json_with_data(self, web_client):
        await _seed_briefing("b-api-1", "Test Q")
        resp = await web_client.get("/api/briefings")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        ids = [b["briefing_id"] for b in data]
        assert "b-api-1" in ids

    async def test_get_briefing_json(self, web_client):
        await _seed_briefing("b-api-2", "Test Q2")
        resp = await web_client.get("/api/briefings/b-api-2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["research_question"] == "Test Q2"

    async def test_get_briefing_json_not_found(self, web_client):
        resp = await web_client.get("/api/briefings/nope")
        assert resp.status_code == 404
        assert resp.json()["error"] == "Not found"
