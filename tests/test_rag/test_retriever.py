"""Tests for hybrid retrieval with RRF."""

import uuid

import pytest

from researchforge.rag.retriever import reciprocal_rank_fusion, retrieve
from researchforge.rag.store import VectorStore

EMBEDDING_DIM = 768


def _make_vector(seed: int = 0) -> list[float]:
    vec = [(seed * (i + 1) % 1000) / 1000.0 for i in range(EMBEDDING_DIM)]
    norm = sum(v**2 for v in vec) ** 0.5
    return [v / (norm or 1.0) for v in vec]


class TestReciprocalRankFusion:
    def test_single_ranking(self):
        result = reciprocal_rank_fusion([["a", "b", "c"]])
        assert result == ["a", "b", "c"]

    def test_two_rankings_agreement(self):
        """When both rankings agree, fused result maintains order."""
        result = reciprocal_rank_fusion([
            ["a", "b", "c"],
            ["a", "b", "c"],
        ])
        assert result[0] == "a"

    def test_two_rankings_disagreement(self):
        """Items appearing in both rankings get boosted."""
        result = reciprocal_rank_fusion([
            ["a", "b", "c"],
            ["c", "d", "a"],
        ])
        # 'a' appears in both, so should be ranked high
        assert "a" in result[:2]
        # 'c' also appears in both
        assert "c" in result[:3]

    def test_empty_ranking(self):
        result = reciprocal_rank_fusion([[]])
        assert result == []


class TestRetrieve:
    @pytest.fixture
    def seeded_store(self, tmp_data_dir):
        store = VectorStore(tmp_data_dir / "lancedb")
        texts = [
            "Retrieval augmented generation combines search with LLMs",
            "Vector databases store embeddings for similarity search",
            "BM25 is a classic keyword-based retrieval algorithm",
            "Neural networks learn representations from data",
            "Transformers use self-attention mechanisms",
        ]
        ids = [str(uuid.uuid4()) for _ in texts]
        vectors = [_make_vector(i) for i in range(len(texts))]

        store.add_chunks(
            chunk_ids=ids,
            texts=texts,
            vectors=vectors,
            document_id="doc-1",
            source_path="/test.md",
            source_type="markdown",
        )
        store.create_fts_index()
        return store

    async def test_retrieve_returns_results(self, seeded_store, mock_embed_query):
        results = await retrieve("retrieval augmented generation", seeded_store, top_k=3)
        assert len(results) <= 3
        assert all("text" in r for r in results)

    async def test_retrieve_no_vector_in_results(self, seeded_store, mock_embed_query):
        results = await retrieve("vector search", seeded_store, top_k=3)
        for r in results:
            assert "vector" not in r, "Raw vectors should be stripped from results"

    async def test_retrieve_respects_top_k(self, seeded_store, mock_embed_query):
        results = await retrieve("search", seeded_store, top_k=2)
        assert len(results) <= 2
