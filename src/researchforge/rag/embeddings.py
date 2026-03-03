"""Ollama embedding client with batching and task-specific prefixes."""

from __future__ import annotations

import httpx
import structlog

from researchforge.config import get_settings

logger = structlog.get_logger()

BATCH_SIZE = 32


async def embed_texts(
    texts: list[str],
    *,
    prefix: str | None = None,
) -> list[list[float]]:
    """Embed a list of texts via the Ollama embeddings API.

    Args:
        texts: Texts to embed.
        prefix: Optional prefix (e.g., "search_document: " or "search_query: ").
                If None, uses the document prefix from config.
    """
    settings = get_settings()
    if prefix is None:
        prefix = settings.retrieval.embedding_prefix_document

    prefixed = [prefix + t for t in texts]
    model = settings.models.embedding
    base_url = settings.ollama.base_url
    timeout = settings.ollama.request_timeout_seconds

    embeddings: list[list[float]] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        for i in range(0, len(prefixed), BATCH_SIZE):
            batch = prefixed[i : i + BATCH_SIZE]
            # Ollama /api/embed supports batch input
            resp = await client.post(
                f"{base_url}/api/embed",
                json={"model": model, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings.extend(data["embeddings"])

    return embeddings


async def embed_query(query: str) -> list[float]:
    """Embed a single query using the query prefix."""
    settings = get_settings()
    prefix = settings.retrieval.embedding_prefix_query
    results = await embed_texts([query], prefix=prefix)
    return results[0]
