"""Tests for the FastAPI app factory and core setup."""

from __future__ import annotations


class TestHealthEndpoint:
    async def test_health_returns_ok(self, web_client):
        resp = await web_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert "ollama_url" in data


class TestStaticFiles:
    async def test_css_served(self, web_client):
        resp = await web_client.get("/static/style.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]

    async def test_htmx_served(self, web_client):
        resp = await web_client.get("/static/htmx.min.js")
        assert resp.status_code == 200

    async def test_missing_static_404(self, web_client):
        resp = await web_client.get("/static/does_not_exist.xyz")
        assert resp.status_code == 404


class TestDashboardPage:
    async def test_dashboard_renders(self, web_client):
        resp = await web_client.get("/")
        assert resp.status_code == 200
        assert "ResearchForge" in resp.text
        assert "Research Question" in resp.text

    async def test_dashboard_has_htmx_form(self, web_client):
        resp = await web_client.get("/")
        assert 'hx-post="/api/research"' in resp.text
