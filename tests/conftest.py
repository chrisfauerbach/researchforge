"""Shared test fixtures for ResearchForge."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure tests use a test config, not the real one
os.environ["RESEARCHFORGE_OLLAMA__BASE_URL"] = "http://localhost:11434"


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for tests."""
    (tmp_path / "lancedb").mkdir()
    return tmp_path


@pytest.fixture
def sample_markdown(tmp_path) -> Path:
    """Create a sample Markdown file for testing."""
    content = """\
# Introduction

This is a document about retrieval augmented generation (RAG).
RAG combines retrieval systems with language models.

## How RAG Works

RAG systems retrieve relevant documents from a corpus and provide them
as context to a language model. This helps ground the model's responses
in factual information.

### Retrieval Component

The retrieval component uses vector similarity search to find relevant
document chunks. Common approaches include dense retrieval using
embedding models and sparse retrieval using BM25.

## Benefits of RAG

RAG reduces hallucination by providing factual grounding. It also allows
the model to access up-to-date information beyond its training data.

## Challenges

Chunking strategy significantly affects retrieval quality. Too-small
chunks lose context; too-large chunks dilute relevance.
"""
    path = tmp_path / "rag_overview.md"
    path.write_text(content)
    return path


@pytest.fixture
def sample_txt(tmp_path) -> Path:
    """Create a sample plain text file for testing."""
    content = "This is a simple plain text document.\n" * 50
    path = tmp_path / "plain.txt"
    path.write_text(content)
    return path


@pytest.fixture
def mock_embeddings():
    """Mock the embed_texts function to return deterministic vectors."""
    dim = 768

    async def fake_embed(texts, *, prefix=None):
        # Return a deterministic vector for each text based on its hash
        vectors = []
        for t in texts:
            h = hash(t) % (2**31)
            vec = [(h * (i + 1) % 1000) / 1000.0 for i in range(dim)]
            # Normalize
            norm = sum(v**2 for v in vec) ** 0.5
            vectors.append([v / norm for v in vec])
        return vectors

    # Patch at the call site (ingest.py imports embed_texts)
    with patch("researchforge.rag.ingest.embed_texts", side_effect=fake_embed):
        yield fake_embed


@pytest.fixture
def mock_embed_query():
    """Mock the embed_query function."""
    dim = 768

    async def fake_query(query):
        h = hash(query) % (2**31)
        vec = [(h * (i + 1) % 1000) / 1000.0 for i in range(dim)]
        norm = sum(v**2 for v in vec) ** 0.5
        return [v / norm for v in vec]

    with patch("researchforge.rag.retriever.embed_query", side_effect=fake_query):
        yield fake_query
