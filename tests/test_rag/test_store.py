"""Tests for the LanceDB vector store."""

import uuid

from researchforge.rag.store import VectorStore

EMBEDDING_DIM = 768


def _make_vector(seed: int = 0) -> list[float]:
    """Create a deterministic unit vector for testing."""
    vec = [(seed * (i + 1) % 1000) / 1000.0 for i in range(EMBEDDING_DIM)]
    norm = sum(v**2 for v in vec) ** 0.5
    return [v / (norm or 1.0) for v in vec]


class TestVectorStore:
    def test_add_and_count(self, tmp_data_dir):
        store = VectorStore(tmp_data_dir / "lancedb")
        count = store.add_chunks(
            chunk_ids=[str(uuid.uuid4())],
            texts=["Hello world"],
            vectors=[_make_vector(1)],
            document_id="doc-1",
            source_path="/test.md",
            source_type="markdown",
        )
        assert count == 1
        assert store.count() == 1

    def test_add_multiple_chunks(self, tmp_data_dir):
        store = VectorStore(tmp_data_dir / "lancedb")
        ids = [str(uuid.uuid4()) for _ in range(5)]
        texts = [f"Chunk {i}" for i in range(5)]
        vectors = [_make_vector(i) for i in range(5)]

        store.add_chunks(
            chunk_ids=ids,
            texts=texts,
            vectors=vectors,
            document_id="doc-1",
            source_path="/test.md",
            source_type="markdown",
        )
        assert store.count() == 5

    def test_vector_search_returns_results(self, tmp_data_dir):
        store = VectorStore(tmp_data_dir / "lancedb")
        ids = [str(uuid.uuid4()) for _ in range(3)]
        texts = ["apple banana", "cherry date", "elderberry fig"]
        vectors = [_make_vector(i) for i in range(3)]

        store.add_chunks(
            chunk_ids=ids,
            texts=texts,
            vectors=vectors,
            document_id="doc-1",
            source_path="/test.md",
            source_type="markdown",
        )

        results = store.vector_search(_make_vector(0), limit=2)
        assert len(results) == 2
        assert "text" in results[0]

    def test_fts_search(self, tmp_data_dir):
        store = VectorStore(tmp_data_dir / "lancedb")
        ids = [str(uuid.uuid4()) for _ in range(3)]
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Natural language processing handles text data",
            "Computer vision processes images and video",
        ]
        vectors = [_make_vector(i) for i in range(3)]

        store.add_chunks(
            chunk_ids=ids,
            texts=texts,
            vectors=vectors,
            document_id="doc-1",
            source_path="/test.md",
            source_type="markdown",
        )
        store.create_fts_index()

        results = store.fts_search("machine learning artificial intelligence", limit=2)
        assert len(results) >= 1
        assert "machine learning" in results[0]["text"].lower()

    def test_content_type_filter(self, tmp_data_dir):
        store = VectorStore(tmp_data_dir / "lancedb")

        # Add source and agent-generated chunks
        store.add_chunks(
            chunk_ids=[str(uuid.uuid4())],
            texts=["Source content"],
            vectors=[_make_vector(1)],
            document_id="doc-1",
            source_path="/source.md",
            source_type="markdown",
            content_type="source",
        )
        store.add_chunks(
            chunk_ids=[str(uuid.uuid4())],
            texts=["Agent generated content"],
            vectors=[_make_vector(2)],
            document_id="doc-2",
            source_path="/briefing.md",
            source_type="markdown",
            content_type="agent_generated",
        )

        # Filter to source only
        results = store.vector_search(
            _make_vector(1),
            limit=10,
            where="content_type = 'source'",
        )
        assert all(r["content_type"] == "source" for r in results)
