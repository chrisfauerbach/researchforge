"""Tests for document chunking."""

from unittest.mock import patch

from researchforge.rag.chunker import Chunk, chunk_document


def _patch_settings(**overrides):
    """Helper to patch chunking settings for tests."""
    from researchforge.config import Settings

    settings = Settings()
    for key, val in overrides.items():
        setattr(settings.chunking, key, val)
    return patch("researchforge.rag.chunker.get_settings", return_value=settings)


class TestChunkDocument:
    def test_plain_text_produces_chunks(self):
        text = "Hello world. " * 200  # ~2600 chars
        with _patch_settings(chunk_size=500, chunk_overlap=50):
            chunks = chunk_document(text, source_type="txt")
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_indices_sequential(self):
        text = "Some text. " * 200
        with _patch_settings(chunk_size=500, chunk_overlap=50):
            chunks = chunk_document(text, source_type="txt")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_markdown_preserves_headers(self):
        text = """\
# Title

Some intro text that is long enough to be a chunk on its own.

## Section One

Content for section one with enough text to fill a chunk.

## Section Two

Content for section two with enough text to fill another chunk.
"""
        with _patch_settings(chunk_size=200, chunk_overlap=20):
            chunks = chunk_document(text, source_type="markdown")

        # At least some chunks should have section metadata
        has_h1 = any(c.metadata.get("section_h1") for c in chunks)
        assert has_h1, "Expected at least one chunk with h1 metadata"

    def test_short_text_single_chunk(self):
        text = "Short document."
        with _patch_settings(chunk_size=1500, chunk_overlap=200):
            chunks = chunk_document(text, source_type="txt")
        assert len(chunks) == 1
        assert chunks[0].text == "Short document."

    def test_extra_metadata_passed_through(self):
        text = "Some text content here."
        with _patch_settings(chunk_size=1500, chunk_overlap=200):
            chunks = chunk_document(
                text,
                source_type="txt",
                extra_metadata={"source_path": "/test.txt"},
            )
        assert chunks[0].metadata["source_path"] == "/test.txt"

    def test_empty_text_no_chunks(self):
        with _patch_settings(chunk_size=1500, chunk_overlap=200):
            chunks = chunk_document("", source_type="txt")
        assert chunks == []

    def test_chunk_size_respected(self):
        text = "Word " * 1000  # ~5000 chars
        with _patch_settings(chunk_size=500, chunk_overlap=50):
            chunks = chunk_document(text, source_type="txt")
        for c in chunks:
            # Allow some flexibility for overlap and splitting
            assert len(c.text) <= 600, f"Chunk too large: {len(c.text)} chars"
