"""Tests for end-to-end evaluation module."""

from __future__ import annotations

from researchforge.eval.e2e_eval import load_reference_briefings


class TestLoadReferenceBriefings:
    def test_loads_markdown_files(self, tmp_path):
        (tmp_path / "topic_one.md").write_text("# Topic One\nContent one.")
        (tmp_path / "topic_two.md").write_text("# Topic Two\nContent two.")

        refs = load_reference_briefings(tmp_path)
        assert len(refs) == 2
        assert refs[0]["topic"] == "Topic One"
        assert "Content one" in refs[0]["text"]
        assert refs[1]["topic"] == "Topic Two"

    def test_empty_directory(self, tmp_path):
        refs = load_reference_briefings(tmp_path)
        assert refs == []

    def test_ignores_non_markdown(self, tmp_path):
        (tmp_path / "notes.txt").write_text("Not markdown")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "real.md").write_text("# Real\nContent")

        refs = load_reference_briefings(tmp_path)
        assert len(refs) == 1
        assert refs[0]["topic"] == "Real"
