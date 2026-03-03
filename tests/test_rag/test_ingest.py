"""Tests for the ingest orchestrator."""

import pytest

from researchforge.db.repository import Repository
from researchforge.rag.ingest import ingest_file
from researchforge.rag.store import VectorStore


class TestIngest:
    @pytest.fixture
    async def repo(self, tmp_data_dir):
        repo = Repository(tmp_data_dir / "metadata.db")
        await repo.initialize()
        yield repo
        await repo.close()

    async def test_ingest_markdown(self, tmp_data_dir, sample_markdown, mock_embeddings, repo):
        store = VectorStore(tmp_data_dir / "lancedb")
        result = await ingest_file(sample_markdown, store, repo)

        assert not result.skipped
        assert result.chunk_count > 0
        assert result.source_type == "markdown"
        assert store.count() == result.chunk_count

    async def test_ingest_txt(self, tmp_data_dir, sample_txt, mock_embeddings, repo):
        store = VectorStore(tmp_data_dir / "lancedb")
        result = await ingest_file(sample_txt, store, repo)

        assert not result.skipped
        assert result.chunk_count > 0
        assert result.source_type == "txt"

    async def test_dedup_skips_second_ingest(
        self, tmp_data_dir, sample_markdown, mock_embeddings, repo
    ):
        store = VectorStore(tmp_data_dir / "lancedb")

        result1 = await ingest_file(sample_markdown, store, repo)
        result2 = await ingest_file(sample_markdown, store, repo)

        assert not result1.skipped
        assert result2.skipped
        assert result2.reason == "duplicate"
        # Should still have the same number of chunks (not doubled)
        assert store.count() == result1.chunk_count

    async def test_ingest_records_metadata(
        self, tmp_data_dir, sample_markdown, mock_embeddings, repo
    ):
        store = VectorStore(tmp_data_dir / "lancedb")
        result = await ingest_file(sample_markdown, store, repo)

        doc = await repo.get_document_by_hash(result.document_id)
        # Check by querying for the actual hash instead
        from researchforge.rag.ingest import _file_hash

        fhash = _file_hash(sample_markdown)
        doc = await repo.get_document_by_hash(fhash)
        assert doc is not None
        assert doc["chunk_count"] == result.chunk_count
