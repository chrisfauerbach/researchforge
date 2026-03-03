# Evaluating Large Language Model Outputs

## Overview

Evaluating LLM outputs is essential for ensuring quality, detecting regressions, and comparing models. Unlike traditional software testing, LLM evaluation must account for the non-deterministic, open-ended nature of generated text. A robust evaluation framework combines automated metrics, LLM-as-judge scoring, and human assessment.

## Evaluation Dimensions

### Structural Quality
Does the output conform to the expected format? For structured outputs (JSON, reports with sections), this can be validated programmatically.

### Relevance
Does the output address the assigned task or question? Measured by comparing the output content against the input requirements.

### Completeness
Are all requested components present? For research briefings, this includes sections like findings, methodology, limitations, and references [1].

### Coherence
Does the output flow logically without contradictions? This measures internal consistency and logical structure.

### Factual Accuracy
Are the claims in the output supported by evidence? In RAG systems, this can be partially automated by checking whether output claims trace back to retrieved source documents.

## Evaluation Methods

### Automated Heuristics
- Word count thresholds (completeness proxy)
- Citation count (attribution proxy)
- Readability scores (Flesch-Kincaid)
- JSON schema validation (structural quality)

### LLM-as-Judge
Use a separate LLM to score outputs against a rubric. Mitigations for reliability include:
- Running evaluations 3 times and taking the median score
- Using structured output (JSON schema) for consistent scoring
- Combining LLM judgment with heuristic checks

### Human Evaluation
The gold standard for quality assessment. Humans rate outputs on specific criteria using a rubric. Used to calibrate automated scoring and catch issues that automated methods miss [2].

## Regression Detection

Track evaluation scores over time and flag any metric that drops more than 10% from the rolling average of the last 5 evaluation runs. This catches quality degradation from model updates, configuration changes, or corpus modifications.

## References

[1] Chang et al., "A Survey on Evaluation of Large Language Models," 2024.
[2] Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS 2023.
