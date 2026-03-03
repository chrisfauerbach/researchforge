"""Tests for retrieval evaluation metrics."""

from __future__ import annotations

import json

from researchforge.eval.retrieval_eval import (
    keyword_hits,
    load_retrieval_test_set,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestKeywordHits:
    def test_all_hits(self):
        texts = ["This discusses RAG and retrieval", "Vector search methods"]
        keywords = ["RAG", "vector"]
        hits = keyword_hits(texts, keywords)
        assert hits == [True, True]

    def test_no_hits(self):
        texts = ["Nothing relevant here", "Also nothing"]
        keywords = ["quantum", "physics"]
        hits = keyword_hits(texts, keywords)
        assert hits == [False, False]

    def test_case_insensitive(self):
        texts = ["This mentions rag techniques"]
        keywords = ["RAG"]
        hits = keyword_hits(texts, keywords)
        assert hits == [True]

    def test_partial_hits(self):
        texts = ["About RAG", "About cats", "About retrieval"]
        keywords = ["RAG", "retrieval"]
        hits = keyword_hits(texts, keywords)
        assert hits == [True, False, True]

    def test_empty_texts(self):
        assert keyword_hits([], ["RAG"]) == []

    def test_empty_keywords(self):
        hits = keyword_hits(["some text"], [])
        assert hits == [False]


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k([True, True, True]) == 1.0

    def test_none_relevant(self):
        assert precision_at_k([False, False, False]) == 0.0

    def test_partial(self):
        assert precision_at_k([True, False, True]) == 2 / 3

    def test_with_k(self):
        assert precision_at_k([True, False, True, True], k=2) == 0.5

    def test_empty(self):
        assert precision_at_k([]) == 0.0


class TestRecallAtK:
    def test_all_found(self):
        assert recall_at_k([True, True], 2) == 1.0

    def test_none_found(self):
        assert recall_at_k([False, False], 2) == 0.0

    def test_partial(self):
        assert recall_at_k([True, False], 3) == 1 / 3

    def test_with_k(self):
        assert recall_at_k([True, True, False, True], 5, k=2) == 2 / 5

    def test_zero_expected(self):
        assert recall_at_k([True, True], 0) == 0.0

    def test_caps_at_one(self):
        # More hits than expected (shouldn't happen but test edge case)
        assert recall_at_k([True, True, True], 2) == 1.0


class TestReciprocalRank:
    def test_first_hit(self):
        assert reciprocal_rank([True, False, False]) == 1.0

    def test_second_hit(self):
        assert reciprocal_rank([False, True, False]) == 0.5

    def test_third_hit(self):
        assert reciprocal_rank([False, False, True]) == 1 / 3

    def test_no_hits(self):
        assert reciprocal_rank([False, False, False]) == 0.0

    def test_empty(self):
        assert reciprocal_rank([]) == 0.0


class TestLoadTestSet:
    def test_loads_jsonl(self, tmp_path):
        data = [
            {"question": "What is RAG?", "expected_keywords": ["RAG"], "topic": "basics"},
            {"question": "How does BM25 work?", "expected_keywords": ["BM25", "ranking"]},
        ]
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        cases = load_retrieval_test_set(path)
        assert len(cases) == 2
        assert cases[0].question == "What is RAG?"
        assert cases[0].expected_keywords == ["RAG"]
        assert cases[0].topic == "basics"
        assert cases[1].topic == ""

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text(
            '{"question": "q1", "expected_keywords": ["k1"]}\n\n'
            '{"question": "q2", "expected_keywords": ["k2"]}\n'
        )
        cases = load_retrieval_test_set(path)
        assert len(cases) == 2
