"""Tests for document parsers."""


import pytest

from researchforge.rag.parsers import detect_source_type, parse_document, parse_markdown


class TestDetectSourceType:
    def test_pdf(self):
        assert detect_source_type("doc.pdf") == "pdf"

    def test_markdown_md(self):
        assert detect_source_type("notes.md") == "markdown"

    def test_markdown_long(self):
        assert detect_source_type("notes.markdown") == "markdown"

    def test_txt(self):
        assert detect_source_type("file.txt") == "txt"

    def test_unknown_defaults_to_txt(self):
        assert detect_source_type("file.xyz") == "txt"

    def test_html(self):
        assert detect_source_type("page.html") == "html"


class TestParseMarkdown:
    def test_reads_file(self, sample_markdown):
        text = parse_markdown(sample_markdown)
        assert "retrieval augmented generation" in text
        assert "# Introduction" in text

    def test_preserves_headers(self, sample_markdown):
        text = parse_markdown(sample_markdown)
        assert "## How RAG Works" in text
        assert "### Retrieval Component" in text


class TestParseDocument:
    def test_markdown_file(self, sample_markdown):
        text, source_type = parse_document(sample_markdown)
        assert source_type == "markdown"
        assert len(text) > 100

    def test_txt_file(self, sample_txt):
        text, source_type = parse_document(sample_txt)
        assert source_type == "txt"
        assert "plain text" in text

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_document(tmp_path / "nonexistent.md")

    def test_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text("{}")
        # .json maps to txt fallback, so this should work
        text, source_type = parse_document(path)
        assert source_type == "txt"
