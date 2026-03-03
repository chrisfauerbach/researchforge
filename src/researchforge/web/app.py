"""FastAPI application — minimal placeholder for Phase 1."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def create_app() -> FastAPI:
    app = FastAPI(title="ResearchForge", version="0.1.0")

    @app.get("/api/health")
    async def health():
        return JSONResponse({"status": "ok", "version": "0.1.0"})

    return app
