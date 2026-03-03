"""FastAPI application factory with lifespan, Jinja2, and static files."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from researchforge.config import get_settings
from researchforge.db.repository import Repository

logger = structlog.get_logger()

_WEB_DIR = Path(__file__).parent
_STATIC_DIR = _WEB_DIR / "static"
_TEMPLATES_DIR = _WEB_DIR / "templates"

# Shared state accessible via app.state
_repo: Repository | None = None


def get_templates() -> Jinja2Templates:
    return Jinja2Templates(directory=str(_TEMPLATES_DIR))


async def get_repo() -> Repository:
    if _repo is None:
        raise RuntimeError("Repository not initialized")
    return _repo


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB. Shutdown: close DB."""
    global _repo
    settings = get_settings()
    _repo = Repository(settings.storage.metadata_db_path)
    await _repo.initialize()
    logger.info("web_startup", db=str(settings.storage.metadata_db_path))
    yield
    if _repo:
        await _repo.close()
        _repo = None
    logger.info("web_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ResearchForge",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Static files
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Health endpoint
    @app.get("/api/health")
    async def health():
        settings = get_settings()
        return JSONResponse({
            "status": "ok",
            "version": "0.1.0",
            "ollama_url": settings.ollama.base_url,
        })

    # Register route modules
    from researchforge.web.routes.briefings import router as briefings_router
    from researchforge.web.routes.corpus import router as corpus_router
    from researchforge.web.routes.eval import router as eval_router
    from researchforge.web.routes.research import router as research_router

    app.include_router(research_router)
    app.include_router(briefings_router)
    app.include_router(corpus_router)
    app.include_router(eval_router)

    return app
