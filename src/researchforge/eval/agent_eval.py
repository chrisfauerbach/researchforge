"""Agent-level evaluation: score individual agent outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from researchforge.eval.judge import LLMJudge, RubricScore

logger = structlog.get_logger()

AGENT_TASK_DESCRIPTIONS = {
    "planner": (
        "Produce a research plan with sub-questions and "
        "search strategies for the given topic."
    ),
    "gatherer": "Retrieve relevant evidence from the corpus for sub-questions.",
    "analyst": (
        "Synthesize gathered evidence into a structured analysis "
        "with key findings and gaps."
    ),
    "critic": "Review the analysis for quality, completeness, and factual accuracy.",
    "writer": "Produce a well-structured research briefing from the validated analysis.",
}


@dataclass
class CriticTestCase:
    """A test case for critic error detection."""

    analysis: str
    planted_errors: list[str]
    error_descriptions: list[str]


@dataclass
class CriticEvalResult:
    """Result of critic error detection evaluation."""

    total_cases: int
    total_errors_planted: int
    total_errors_detected: int
    detection_rate: float
    case_details: list[dict] = field(default_factory=list)


@dataclass
class AgentEvalResult:
    """Result of evaluating a single agent role."""

    agent_name: str
    case_count: int
    mean_score: float
    scores: list[RubricScore] = field(default_factory=list)
    critic_eval: CriticEvalResult | None = None


def load_critic_test_set(path: str | Path) -> list[CriticTestCase]:
    """Load critic test cases from a JSONL file."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(CriticTestCase(
                analysis=data["analysis"],
                planted_errors=data.get("planted_errors", []),
                error_descriptions=data.get("error_descriptions", []),
            ))
    return cases


async def evaluate_critic(
    test_set_path: str | Path,
    model: str | None = None,
) -> CriticEvalResult:
    """Evaluate the critic agent's ability to detect planted errors.

    Feeds analyses with known errors to the critic and checks whether
    the critic identifies them (by returning a 'revise' verdict).
    """
    from researchforge.agents.ollama_client import ollama_chat
    from researchforge.config import get_settings

    settings = get_settings()
    critic_model = model or settings.models.critic
    cases = load_critic_test_set(test_set_path)

    total_errors = 0
    detected = 0
    case_details = []

    for case in cases:
        total_errors += len(case.planted_errors)

        try:
            result = await ollama_chat(
                model=critic_model,
                system_prompt=(
                    "You are a research quality critic. Review the following analysis "
                    "for errors, inaccuracies, and logical problems. "
                    "Respond with JSON: {\"verdict\": \"pass\" or \"revise\", "
                    "\"issues\": [list of issues found]}"
                ),
                user_message=case.analysis,
                expect_json=True,
                agent_name="critic_eval",
            )
            parsed = result.get("parsed", {})
            verdict = parsed.get("verdict", "pass")
            issues = parsed.get("issues", [])

            # Count detection: if critic said "revise", it detected something
            case_detected = len(case.planted_errors) if verdict == "revise" else 0
            detected += case_detected

            case_details.append({
                "planted_errors": case.planted_errors,
                "verdict": verdict,
                "issues_found": issues,
                "detected": case_detected > 0,
            })
        except Exception as exc:
            logger.warning("critic_eval_error", error=str(exc))
            case_details.append({
                "planted_errors": case.planted_errors,
                "verdict": "error",
                "issues_found": [],
                "detected": False,
            })

    detection_rate = detected / total_errors if total_errors > 0 else 0.0

    return CriticEvalResult(
        total_cases=len(cases),
        total_errors_planted=total_errors,
        total_errors_detected=detected,
        detection_rate=round(detection_rate, 4),
        case_details=case_details,
    )


async def evaluate_agent(
    agent_name: str,
    briefings: list[dict],
    judge: LLMJudge | None = None,
) -> AgentEvalResult:
    """Evaluate an agent role by scoring its outputs from completed briefings.

    Args:
        agent_name: Agent role name (planner, gatherer, analyst, critic, writer).
        briefings: List of completed briefing dicts from the DB.
        judge: LLMJudge instance (created if not provided).
    """
    if judge is None:
        judge = LLMJudge()

    task_desc = AGENT_TASK_DESCRIPTIONS.get(agent_name, f"Perform the {agent_name} role.")
    scores: list[RubricScore] = []

    for briefing in briefings:
        text = briefing.get("briefing_markdown", "")
        if not text.strip():
            continue

        try:
            score = await judge.score(text, task_description=task_desc)
            scores.append(score)
        except Exception as exc:
            logger.warning("agent_eval_error", agent=agent_name, error=str(exc))

    mean = sum(s.weighted_total for s in scores) / len(scores) if scores else 0.0

    return AgentEvalResult(
        agent_name=agent_name,
        case_count=len(scores),
        mean_score=round(mean, 4),
        scores=scores,
    )
