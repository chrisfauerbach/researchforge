"""Tests for agent evaluation module."""

from __future__ import annotations

import json

from researchforge.eval.agent_eval import load_critic_test_set


class TestLoadCriticTestSet:
    def test_loads_cases(self, tmp_path):
        data = [
            {
                "analysis": "Bad analysis with errors.",
                "planted_errors": ["error1"],
                "error_descriptions": ["Description of error1"],
            },
            {
                "analysis": "Another bad analysis.",
                "planted_errors": ["error2", "error3"],
                "error_descriptions": ["Desc 2", "Desc 3"],
            },
        ]
        path = tmp_path / "critic_test.jsonl"
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        cases = load_critic_test_set(path)
        assert len(cases) == 2
        assert cases[0].analysis == "Bad analysis with errors."
        assert cases[0].planted_errors == ["error1"]
        assert len(cases[1].planted_errors) == 2

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text(
            '{"analysis": "a1", "planted_errors": ["e1"], "error_descriptions": ["d1"]}\n\n'
        )
        cases = load_critic_test_set(path)
        assert len(cases) == 1
