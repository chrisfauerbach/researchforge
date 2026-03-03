"""End-to-end evaluation: run pipeline on reference topics, compare to references."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog

from researchforge.eval.judge import LLMJudge, RubricScore

logger = structlog.get_logger()

E2E_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator comparing a generated research briefing against a reference.
Score the generated briefing on these criteria (0.0 to 1.0 each):

1. structural_validity: Well-structured with clear sections, formatting matches reference quality.
2. relevance: Covers the same core topics and questions as the reference.
3. completeness: Addresses all key points present in the reference.
4. coherence: Logical flow, no contradictions, ideas connect smoothly.
5. conciseness: Information-dense, no unnecessary padding.

Respond ONLY with valid JSON."""


@dataclass
class E2ECaseResult:
    """Result of a single end-to-end test case."""

    topic: str
    reference_file: str
    score: RubricScore | None = None
    pipeline_status: str = ""
    error: str = ""


@dataclass
class E2EEvalResult:
    """Aggregated end-to-end evaluation results."""

    case_count: int
    mean_score: float
    cases: list[E2ECaseResult] = field(default_factory=list)


def load_reference_briefings(dir_path: str | Path) -> list[dict]:
    """Load reference briefings from a directory of Markdown files.

    Returns list of {topic, text, path} dicts.
    """
    dir_path = Path(dir_path)
    refs = []
    for md_file in sorted(dir_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        topic = md_file.stem.replace("_", " ").title()
        refs.append({"topic": topic, "text": text, "path": str(md_file)})
    return refs


async def evaluate_e2e(
    reference_dir: str | Path,
    judge: LLMJudge | None = None,
    depth: str = "quick",
) -> E2EEvalResult:
    """Run end-to-end evaluation against reference briefings.

    For each reference, runs the pipeline on the same topic and scores
    the generated output against the reference using LLM-as-judge.
    """
    from researchforge.agents.graph import run_pipeline

    if judge is None:
        judge = LLMJudge()

    references = load_reference_briefings(reference_dir)
    cases: list[E2ECaseResult] = []

    for ref in references:
        case = E2ECaseResult(topic=ref["topic"], reference_file=ref["path"])

        try:
            # Run pipeline on the reference topic
            question = f"Provide a comprehensive overview of: {ref['topic']}"
            final_state = await run_pipeline(question, depth=depth)
            generated = final_state.get("briefing", "")
            case.pipeline_status = final_state.get("status", "unknown")

            if not generated.strip():
                case.error = "Empty briefing generated"
                cases.append(case)
                continue

            # Score generated vs reference
            task_desc = (
                f"Generate a research briefing on '{ref['topic']}' that matches "
                f"the quality and coverage of this reference:\n\n{ref['text'][:2000]}"
            )
            score = await judge.score(generated, task_description=task_desc)
            case.score = score

        except Exception as exc:
            logger.error("e2e_eval_error", topic=ref["topic"], error=str(exc))
            case.error = str(exc)
            case.pipeline_status = "error"

        cases.append(case)

    scored = [c for c in cases if c.score is not None]
    mean = sum(c.score.weighted_total for c in scored) / len(scored) if scored else 0.0

    return E2EEvalResult(
        case_count=len(cases),
        mean_score=round(mean, 4),
        cases=cases,
    )
