"""Extended parser tests for HTML and DOCX formats."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from researchforge.rag.parsers import parse_document, parse_html


class TestParseHtml:
    def test_basic_html(self, tmp_path):
        html = "<html><body><h1>Title</h1><p>Hello world.</p></body></html>"
        path = tmp_path / "test.html"
        path.write_text(html)
        text = parse_html(path)
        assert "Title" in text
        assert "Hello world." in text

    def test_strips_scripts(self, tmp_path):
        html = "<html><body><script>alert('xss')</script><p>Content.</p></body></html>"
        path = tmp_path / "test.html"
        path.write_text(html)
        text = parse_html(path)
        assert "alert" not in text
        assert "Content." in text

    def test_strips_style_nav_footer_header(self, tmp_path):
        html = (
            "<html><body>"
            "<header>Site Header</header>"
            "<nav>Menu</nav>"
            "<main><p>Main content here.</p></main>"
            "<footer>Copyright</footer>"
            "<style>.cls { color: red; }</style>"
            "</body></html>"
        )
        path = tmp_path / "test.html"
        path.write_text(html)
        text = parse_html(path)
        assert "Main content here." in text
        assert "Site Header" not in text
        assert "Menu" not in text
        assert "Copyright" not in text
        assert ".cls" not in text

    def test_handles_encoding(self, tmp_path):
        html = "<html><body><p>Caf\u00e9 na\u00efve r\u00e9sum\u00e9</p></body></html>"
        path = tmp_path / "test.html"
        path.write_text(html, encoding="utf-8")
        text = parse_html(path)
        assert "Caf\u00e9" in text

    def test_parse_document_html(self, tmp_path):
        html = "<html><body><p>Test content</p></body></html>"
        path = tmp_path / "page.html"
        path.write_text(html)
        text, source_type = parse_document(path)
        assert source_type == "html"
        assert "Test content" in text

    def test_parse_document_htm(self, tmp_path):
        html = "<html><body><p>HTM content</p></body></html>"
        path = tmp_path / "page.htm"
        path.write_text(html)
        text, source_type = parse_document(path)
        assert source_type == "html"
        assert "HTM content" in text


class TestParseDocx:
    @patch("mammoth.convert_to_markdown")
    def test_basic_docx(self, mock_convert, tmp_path):
        mock_result = MagicMock()
        mock_result.value = "# Document Title\n\nSome paragraph text."
        mock_convert.return_value = mock_result

        path = tmp_path / "test.docx"
        path.write_bytes(b"fake docx content")

        from researchforge.rag.parsers import parse_docx

        text = parse_docx(path)
        assert "Document Title" in text
        assert "Some paragraph text." in text

    @patch("mammoth.convert_to_markdown")
    def test_parse_document_docx(self, mock_convert, tmp_path):
        mock_result = MagicMock()
        mock_result.value = "Extracted markdown content."
        mock_convert.return_value = mock_result

        path = tmp_path / "report.docx"
        path.write_bytes(b"fake docx content")

        text, source_type = parse_document(path)
        assert source_type == "docx"
        assert "Extracted markdown content." in text
