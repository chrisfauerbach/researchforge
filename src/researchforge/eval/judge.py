"""LLM-as-Judge: score agent outputs against a rubric using Ollama."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import structlog

from researchforge.config import get_settings

logger = structlog.get_logger()

# Rubric criteria with weights
RUBRIC_CRITERIA = {
    "structural_validity": 0.15,
    "relevance": 0.30,
    "completeness": 0.25,
    "coherence": 0.20,
    "conciseness": 0.10,
}

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator. Score the given text on these criteria (0.0 to 1.0 each):

1. structural_validity (0.15 weight): Output is well-structured with clear sections/formatting.
2. relevance (0.30 weight): Content directly addresses the assigned task or question.
3. completeness (0.25 weight): All requested components are present and adequately covered.
4. coherence (0.20 weight): Logical flow, no contradictions, ideas connect smoothly.
5. conciseness (0.10 weight): No unnecessary repetition or padding. Information-dense.

Respond ONLY with valid JSON matching this exact schema."""

JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "structural_validity": {"type": "number"},
        "relevance": {"type": "number"},
        "completeness": {"type": "number"},
        "coherence": {"type": "number"},
        "conciseness": {"type": "number"},
    },
    "required": list(RUBRIC_CRITERIA.keys()),
}


@dataclass
class RubricScore:
    """Scores for a single evaluation."""

    structural_validity: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    coherence: float = 0.0
    conciseness: float = 0.0
    weighted_total: float = 0.0
    heuristic_scores: dict = field(default_factory=dict)

    def compute_weighted_total(self) -> float:
        total = 0.0
        for criterion, weight in RUBRIC_CRITERIA.items():
            total += getattr(self, criterion, 0.0) * weight
        self.weighted_total = round(total, 4)
        return self.weighted_total


# --- Heuristic checks (no LLM needed) ---

def heuristic_word_count(text: str) -> float:
    """Score based on word count. >200 words = 1.0, scaled below."""
    count = len(text.split())
    if count >= 200:
        return 1.0
    return round(count / 200, 2)


def heuristic_citation_count(text: str) -> float:
    """Score based on presence of citations."""
    citations = len(re.findall(r"\[\d+\]|\[Source", text))
    if citations >= 3:
        return 1.0
    if citations >= 1:
        return 0.6
    return 0.0


def heuristic_section_count(text: str) -> float:
    """Score based on markdown section headers."""
    headers = len(re.findall(r"^#{1,3}\s", text, re.MULTILINE))
    if headers >= 4:
        return 1.0
    if headers >= 2:
        return 0.7
    if headers >= 1:
        return 0.4
    return 0.0


def heuristic_readability(text: str) -> float:
    """Simple readability proxy: avg sentence length. Shorter = more readable."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    # Ideal: 15-25 words per sentence
    if 10 <= avg_words <= 30:
        return 1.0
    if avg_words < 10:
        return 0.6
    return max(0.2, 1.0 - (avg_words - 30) * 0.02)


def compute_heuristic_scores(text: str) -> dict[str, float]:
    """Compute all heuristic quality checks."""
    return {
        "word_count": heuristic_word_count(text),
        "citations": heuristic_citation_count(text),
        "sections": heuristic_section_count(text),
        "readability": heuristic_readability(text),
    }


# --- LLM Judge ---

class LLMJudge:
    """Scores text against a rubric using an Ollama model."""

    def __init__(self, model: str | None = None, num_runs: int = 3):
        self.model = model or get_settings().models.eval_judge
        self.num_runs = num_runs

    async def score(self, text: str, task_description: str = "") -> RubricScore:
        """Score text using LLM-as-judge with median of multiple runs.

        Args:
            text: The text to evaluate.
            task_description: Optional context about what the text should accomplish.
        """
        from researchforge.agents.ollama_client import ollama_chat

        user_msg = f"Task: {task_description}\n\n---\n\nText to evaluate:\n{text}"

        run_scores: list[dict[str, float]] = []

        for run_idx in range(self.num_runs):
            try:
                result = await ollama_chat(
                    model=self.model,
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    user_message=user_msg,
                    expect_json=True,
                    json_schema=JUDGE_JSON_SCHEMA,
                    agent_name="eval_judge",
                )
                parsed = result.get("parsed", {})
                if parsed and all(k in parsed for k in RUBRIC_CRITERIA):
                    # Clamp values to 0-1
                    clamped = {
                        k: max(0.0, min(1.0, float(parsed[k])))
                        for k in RUBRIC_CRITERIA
                    }
                    run_scores.append(clamped)
                else:
                    logger.warning("judge_incomplete_response", run=run_idx)
            except Exception as exc:
                logger.warning("judge_run_failed", run=run_idx, error=str(exc))

        if not run_scores:
            logger.error("judge_all_runs_failed")
            score = RubricScore()
            score.heuristic_scores = compute_heuristic_scores(text)
            return score

        # Take median across runs for each criterion
        median_scores = {}
        for criterion in RUBRIC_CRITERIA:
            values = [s[criterion] for s in run_scores]
            median_scores[criterion] = round(statistics.median(values), 4)

        score = RubricScore(**median_scores)
        score.heuristic_scores = compute_heuristic_scores(text)
        score.compute_weighted_total()
        return score

    async def score_heuristic_only(self, text: str) -> RubricScore:
        """Score using only heuristic checks (no LLM call)."""
        heuristics = compute_heuristic_scores(text)

        # Map heuristics to rubric criteria as approximations
        score = RubricScore(
            structural_validity=heuristics["sections"],
            relevance=0.5,  # Cannot assess without LLM
            completeness=heuristics["word_count"],
            coherence=heuristics["readability"],
            conciseness=min(1.0, heuristics["readability"]),
            heuristic_scores=heuristics,
        )
        score.compute_weighted_total()
        return score
