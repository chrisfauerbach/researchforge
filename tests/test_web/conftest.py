"""Shared fixtures for web tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

import researchforge.web.app as app_module
from researchforge.db.repository import Repository
from researchforge.web.app import create_app


@pytest.fixture
async def web_client(tmp_path):
    """Create a test client with a real temporary database.

    Manually initializes the repository since httpx ASGITransport
    does not invoke ASGI lifespan events.
    """
    db_path = str(tmp_path / "test_metadata.db")

    repo = Repository(db_path)
    await repo.initialize()

    # Inject the repo into the app module so get_repo() works
    original_repo = app_module._repo
    app_module._repo = repo

    with patch("researchforge.web.app.get_settings") as mock_settings:
        settings = mock_settings.return_value
        settings.storage.metadata_db_path = db_path
        settings.ollama.base_url = "http://localhost:11434"
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as ac:
            yield ac

    await repo.close()
    app_module._repo = original_repo
