# ResearchForge — Implementation Plan

> Phased delivery plan for building the ResearchForge platform.
> Each phase produces a working (if limited) system.
> Version: 0.1.0-draft | Last updated: 2026-03-03

---

## Overview

The project is broken into **5 phases**. Each phase builds on the previous one. Phase 1 is a vertical slice — the minimum end-to-end system. Later phases add agents, the MCP server, eval engine, and polish.

```
Phase 1: Vertical Slice          [S/M]  ← Single agent + RAG + CLI
Phase 2: Full Agent Pipeline     [L]    ← All 5 agents + LangGraph orchestration
Phase 3: Web UI                  [M]    ← FastAPI + HTMX dashboard
Phase 4: MCP Server + Corpus     [M]    ← MCP tools + feedback loop
Phase 5: Eval Engine + Polish    [L]    ← Eval suite + benchmarking + hardening
```

**Total estimated relative sizing: ~XL (sum of all phases)**

---

## Phase 1 — Vertical Slice

**Goal:** A single agent that takes a research question, retrieves relevant context from a local RAG corpus, and produces a basic briefing. End to end, from document ingestion to briefing output.

**Complexity: S/M**

### What's Built

| Component | Scope |
|-----------|-------|
| **Docker infrastructure** | `docker-compose.yml` (GPU Ollama + app), `Dockerfile`, `.dockerignore`, CPU override |
| **RAG ingestion** | Parse PDFs and Markdown → chunk → embed → store in LanceDB |
| **RAG retrieval** | Hybrid search (vector + BM25) via LanceDB |
| **Single "Researcher" agent** | One LLM call that receives the question + retrieved context and produces a briefing |
| **CLI interface** | `docker compose exec app python -m researchforge ingest <path>` and `research "<question>"` |
| **SQLite metadata** | Document and chunk tracking |
| **Configuration** | `config.yaml` with model and chunking settings |
| **Project skeleton** | pyproject.toml, directory structure, basic logging |

### Specific Deliverables

1. **Docker infrastructure**
   - `docker-compose.yml` — Two services: `ollama` (GPU-enabled, named volume for models) and `app` (built from Dockerfile, bind mounts for data/logs/config)
   - `docker-compose.cpu.yml` — Override that removes GPU reservation for CPU-only hosts
   - `Dockerfile` — Python 3.12-slim base, installs dependencies, copies source, exposes port 8000
   - `.dockerignore` — Excludes `data/`, `logs/`, `.git`, `__pycache__`, `.venv`
   - `scripts/setup_models.sh` — Pulls all required Ollama models inside the ollama container
   - Verify: `docker compose up -d` starts both containers; `docker compose exec ollama nvidia-smi` shows GPU; `docker compose exec ollama ollama list` shows pulled models

2. **`src/researchforge/rag/parsers.py`**
   - `parse_pdf(path) -> str` — uses pymupdf4llm to extract Markdown from PDF
   - `parse_markdown(path) -> str` — reads Markdown files
   - Unit tests with a sample PDF and a sample Markdown file

2. **`src/researchforge/rag/chunker.py`**
   - `chunk_document(text, metadata) -> list[Chunk]` — RecursiveCharacterTextSplitter with MarkdownHeaderTextSplitter pre-pass
   - Default: 1500 chars, 200 overlap
   - Unit tests verifying chunk sizes, overlap, and metadata preservation

3. **`src/researchforge/rag/embeddings.py`**
   - `embed_texts(texts: list[str]) -> list[list[float]]` — calls Ollama embeddings API with `nomic-embed-text`
   - Handles batching (32 texts per call)
   - Adds `search_document:` / `search_query:` prefixes appropriately
   - Integration test (requires running Ollama)

4. **`src/researchforge/rag/store.py`**
   - `VectorStore` class wrapping LanceDB
   - `add_chunks(chunks)` — insert chunks + vectors
   - `create_fts_index()` — build Tantivy FTS index on text column
   - Schema matches the DESIGN.md LanceDB schema

5. **`src/researchforge/rag/retriever.py`**
   - `retrieve(query, top_k=5, source_only=False) -> list[Chunk]` — hybrid search with RRF
   - Integration test with a seeded corpus

6. **`src/researchforge/rag/ingest.py`**
   - `ingest_file(path) -> IngestResult` — orchestrates parse → chunk → embed → store
   - Deduplication via file hash (skip if already ingested)
   - Records document metadata in SQLite

7. **`src/researchforge/agents/researcher.py`** (temporary single-agent, replaced in Phase 2)
   - Takes question + retrieved chunks → produces Markdown briefing
   - Single Ollama `/api/chat` call with a system prompt + context
   - Uses `qwen2.5:14b` (or `qwen2.5:7b` on 16GB)

8. **`src/researchforge/__main__.py`**
   - CLI commands via argparse:
     - `ingest <path>` — ingest a file or directory
     - `research "<question>"` — run retrieval + single agent → print briefing
     - `search "<query>"` — search the corpus and print results

9. **`src/researchforge/config.py`**
   - Pydantic Settings loading from `config.yaml` with sensible defaults

10. **`src/researchforge/db/models.py` + `repository.py`**
    - SQLite tables: `documents`, `briefings`, `chunk_lineage`
    - Async access via aiosqlite

### What's Testable

- [x] Ingest a PDF → verify chunks appear in LanceDB with correct metadata
- [x] Ingest a Markdown file → verify structure-aware chunking preserves headers
- [x] Search the corpus → verify hybrid search returns relevant chunks
- [x] Run `research "What is retrieval augmented generation?"` against an ingested RAG paper → verify a coherent briefing is produced
- [x] Run `ingest` on the same file twice → verify dedup (no duplicate chunks)

### Demo

```bash
# Start the full stack (GPU-enabled)
docker compose up -d

# Pull models (first time only)
docker compose exec ollama bash scripts/setup_models.sh

# Ingest some documents (copy files into container or mount a directory)
docker compose exec app python -m researchforge ingest /app/data/sample_docs/rag_survey.pdf

# Search the corpus
docker compose exec app python -m researchforge search "hybrid retrieval methods"

# Run a research query
docker compose exec app python -m researchforge research \
  "What are the main approaches to hybrid search in RAG systems?"
# → Outputs a Markdown briefing to stdout

# CPU-only host:
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

### Testing Strategy

| Layer | Type | Requires Ollama? |
|-------|------|-------------------|
| Docker Compose up | Smoke | No (just checks containers start) |
| GPU passthrough | Smoke | Yes (nvidia-smi in container) |
| Parsers | Unit | No |
| Chunker | Unit | No |
| Embeddings | Integration | Yes |
| VectorStore | Integration | No (mock embeddings) |
| Retriever | Integration | No (mock embeddings) |
| Ingest pipeline | Integration | Yes |
| Single agent | Integration | Yes |
| CLI | End-to-end | Yes |

Tests that require Ollama are marked with `@pytest.mark.ollama` and run inside the app container where the Ollama service is reachable at `http://ollama:11434`.

```bash
# Run unit tests (no Ollama needed)
docker compose exec app pytest -m "not ollama"

# Run all tests (Ollama container must be healthy)
docker compose exec app pytest
```

### Dependencies

- None — this is the first phase.

---

## Phase 2 — Full Agent Pipeline

**Goal:** Replace the single-agent research flow with the full 5-agent pipeline (Planner → Gatherer → Analyst → Critic → Writer) orchestrated by LangGraph.

**Complexity: L**

**Depends on: Phase 1**

### What's Built

| Component | Scope |
|-----------|-------|
| **LangGraph pipeline** | StateGraph with 5 agent nodes + conditional Critic→Analyst edge |
| **PipelineState** | Typed state dict flowing through the graph |
| **5 agent modules** | Planner, Gatherer, Analyst, Critic, Writer — each with its own model and prompt |
| **Prompt templates** | External text files in `agents/prompts/` |
| **Retry + fallback logic** | Structural retry, model fallback, graceful degradation |
| **State checkpointing** | LangGraph SqliteSaver for crash recovery |
| **Structured logging** | structlog → JSON lines with pipeline_id correlation |
| **Pipeline tracing** | Timestamped trace of all agent actions in state |

### Specific Deliverables

1. **`src/researchforge/agents/state.py`**
   - `PipelineState` TypedDict as specified in DESIGN.md

2. **`src/researchforge/agents/graph.py`**
   - LangGraph `StateGraph` definition
   - Nodes: planner, gatherer, analyst, critic, writer
   - Edges: planner→gatherer→analyst→critic→(conditional: writer or analyst)
   - `SqliteSaver` checkpointer
   - `run_pipeline(question, depth="standard") -> PipelineState` entry point
   - `astream_pipeline(question) -> AsyncIterator[event]` for streaming

3. **`src/researchforge/agents/planner.py`**
   - Input: research question
   - Output: `research_plan` (sub-questions, info needs, priorities) as structured JSON
   - Model: `deepseek-r1:14b`
   - Uses Ollama `format: {json_schema}` for output enforcement

4. **`src/researchforge/agents/gatherer.py`**
   - Input: research plan
   - Output: `evidence` (retrieved chunks per sub-question) + `gaps` (info not found)
   - Model: `qwen2.5:7b`
   - Calls `retriever.retrieve()` for each sub-question (parallel via asyncio.gather)

5. **`src/researchforge/agents/analyst.py`**
   - Input: evidence + original question
   - Output: `analysis` (findings, cross-references, contradictions, confidence levels)
   - Model: `qwen2.5:14b`

6. **`src/researchforge/agents/critic.py`**
   - Input: analysis + evidence
   - Output: `critic_verdict` ("pass"/"revise") + `critic_issues`
   - Model: `deepseek-r1:7b`

7. **`src/researchforge/agents/writer.py`**
   - Input: reviewed analysis + evidence + critic issues (if any)
   - Output: `briefing` (Markdown with executive summary, findings, citations, caveats)
   - Model: `mistral-nemo:12b`

8. **Prompt templates** in `src/researchforge/agents/prompts/`
   - One `.txt` file per agent with Jinja2 placeholders for dynamic content
   - Follow the prompt design guidelines from DESIGN.md

9. **Updated CLI**
   - `python -m researchforge research "<question>"` now runs the full pipeline
   - `--depth quick` flag skips Critic (Planner→Gatherer→Analyst→Writer)
   - `--verbose` flag prints agent-by-agent progress to stderr

10. **Updated `config.yaml`**
    - Per-agent model assignment
    - Fallback models
    - Max retries, timeout settings

### What's Testable

- [x] Planner produces valid JSON research plan for a given question
- [x] Gatherer retrieves relevant chunks for each sub-question
- [x] Analyst produces structured analysis from evidence
- [x] Critic correctly identifies planted errors in a bad analysis (use a crafted test input)
- [x] Critic passes a well-formed analysis
- [x] Writer produces coherent Markdown briefing with citations
- [x] Full pipeline end-to-end: question → briefing (on a seeded corpus)
- [x] Critic rejection triggers Analyst re-run (verify via pipeline trace)
- [x] Max retry limit is respected (after 2 rejections, pipeline proceeds to Writer)
- [x] Model fallback works (simulate primary model timeout)
- [x] Pipeline state is checkpointed (kill and resume test)

### Demo

```bash
# Full pipeline with progress
docker compose exec app python -m researchforge research \
  "What are the security implications of RAG systems?" --verbose

# Output:
# [Planner] Decomposing into 5 sub-questions... (deepseek-r1:14b, GPU)
# [Gatherer] Retrieving evidence for sub-question 1/5...
# [Gatherer] Retrieving evidence for sub-question 2/5...
# ...
# [Analyst] Synthesizing findings...
# [Critic] Reviewing analysis... PASS
# [Writer] Producing briefing...
#
# === Research Briefing ===
# # Security Implications of RAG Systems
# ## Executive Summary
# ...

# Quick mode (skip critic)
docker compose exec app python -m researchforge research \
  "What is prompt injection?" --depth quick
```

### Testing Strategy

| Test | Type | Requires Ollama? |
|------|------|-------------------|
| State schema validation | Unit | No |
| Graph topology (edges, conditions) | Unit | No (mock agent nodes) |
| Individual agent prompts produce valid output | Integration | Yes |
| Critic retry loop | Integration | Yes (or mock with deterministic responses) |
| Full pipeline | End-to-end | Yes |
| Checkpointing / resume | Integration | Yes |

---

## Phase 3 — Web UI

**Goal:** Build a local web dashboard for submitting research questions, viewing agent progress in real-time, browsing the corpus, and reading briefings.

**Complexity: M**

**Depends on: Phase 2**

### What's Built

| Component | Scope |
|-----------|-------|
| **FastAPI app** | HTTP server with API routes + Jinja2 templates |
| **SSE streaming** | Real-time agent progress events |
| **HTMX frontend** | Server-rendered pages with dynamic updates |
| **Research dashboard** | Submit questions, view active/completed pipelines |
| **Briefing viewer** | Read briefings with source citations |
| **Corpus browser** | Search + browse ingested documents |
| **Document upload** | Drag-and-drop file ingestion |
| **Configuration page** | View/edit model assignments and settings |

### Specific Deliverables

1. **`src/researchforge/web/app.py`**
   - FastAPI application factory
   - Lifespan handler (startup: initialize DB, verify Ollama; shutdown: cancel tasks)
   - Static file serving (HTMX, CSS)
   - CORS disabled (local-only)

2. **`src/researchforge/web/routes/research.py`**
   - `POST /api/research` — start pipeline, return job_id
   - `GET /api/pipelines/{job_id}/stream` — SSE event stream
   - `GET /api/pipelines` — list active + recent pipelines
   - `GET /` — research dashboard HTML

3. **`src/researchforge/web/routes/briefings.py`**
   - `GET /api/briefings` — list briefings (JSON)
   - `GET /api/briefings/{id}` — get briefing content (JSON)
   - `GET /briefings` — briefings list HTML
   - `GET /briefings/{id}` — briefing viewer HTML

4. **`src/researchforge/web/routes/corpus.py`**
   - `GET /api/corpus/search?q=...` — hybrid search (JSON)
   - `POST /api/corpus/ingest` — upload and ingest file
   - `GET /api/corpus/stats` — corpus statistics
   - `GET /corpus` — corpus browser HTML

5. **`src/researchforge/web/routes/config.py`**
   - `GET /api/config` — current configuration
   - `PUT /api/config` — update configuration (write to config.yaml)
   - `GET /config` — configuration panel HTML

6. **HTML templates** in `src/researchforge/web/templates/`
   - `base.html` — layout with nav sidebar (HTMX loaded)
   - `dashboard.html` — research input + active pipeline cards with SSE
   - `briefing.html` — rendered Markdown + source citations + pipeline trace
   - `corpus.html` — search box + results + upload zone
   - `config.html` — form-based configuration editor

7. **`src/researchforge/web/static/`**
   - `htmx.min.js` (vendored, ~14KB)
   - `style.css` — clean, minimal CSS (no framework needed; CSS variables for theming)

8. **Pipeline event system**
   - `PipelineEventBus` — asyncio-based pub/sub for pipeline events
   - Agents emit events: `stage_start`, `stage_complete`, `error`, `pipeline_complete`
   - SSE endpoint subscribes to events for a specific job_id

### What's Testable

- [x] `POST /api/research` returns job_id and starts pipeline
- [x] SSE stream delivers events as pipeline progresses
- [x] Briefing viewer renders Markdown correctly with citations
- [x] Corpus search returns relevant results
- [x] File upload ingests document and confirms chunk count
- [x] Configuration changes persist to config.yaml
- [x] Health check endpoint returns accurate system status
- [x] UI renders correctly in Chrome/Firefox (manual check)

### Demo

```bash
# Web UI is already running via docker compose up (default CMD is "serve")
# Open http://localhost:8000 in browser
# 1. Type a research question, click "Start Research"
# 2. Watch agent progress bars update in real-time
# 3. Click completed briefing to read it
# 4. Navigate to Corpus tab, search for a term
# 5. Upload a new PDF via the upload zone

# If starting fresh:
docker compose up -d
# → app container starts web server on port 8000
# → ollama container runs GPU-accelerated inference
```

### Testing Strategy

| Test | Type | Requires Ollama? |
|------|------|-------------------|
| API routes (JSON responses) | Unit/Integration | No (mock pipeline) |
| SSE streaming | Integration | No (mock event bus) |
| File upload + ingestion | Integration | Yes |
| Template rendering | Unit | No |
| Full flow: submit → stream → view briefing | End-to-end | Yes |

---

## Phase 4 — MCP Server + Corpus Feedback Loop

**Goal:** Expose ResearchForge as an MCP server so other AI tools can use it, and implement the briefing→corpus feedback loop.

**Complexity: M**

**Depends on: Phase 3**

### What's Built

| Component | Scope |
|-----------|-------|
| **MCP server** | 6 tools: research, query_corpus, ingest_document, list_briefings, get_briefing, get_status |
| **MCP transports** | stdio (for Claude Desktop) and SSE (for HTTP clients) |
| **Corpus feedback loop** | Completed briefings scored → ingested into RAG corpus if quality >= threshold |
| **Quality gate** | Simple heuristic scoring (pre-eval engine) + configurable threshold |
| **Content type filtering** | `source_only` flag in retrieval to exclude agent-generated content |
| **Additional parsers** | HTML and DOCX parsing (parsers.py expanded) |

### Specific Deliverables

1. **`src/researchforge/mcp_server/server.py`**
   - `FastMCP("researchforge")` with all 6 tools as defined in DESIGN.md
   - Long-running research handled via async job pattern (return job_id, poll status)
   - Progress reporting via `ctx.report_progress()` for supported MCP clients

2. **MCP configuration entry point**
   - `docker compose exec -T app python -m researchforge mcp` — starts MCP server (stdio transport)
   - `scripts/mcp_wrapper.sh` — Host-side wrapper script for Claude Desktop integration
   - Claude Desktop config snippet in README:
     ```json
     {
       "mcpServers": {
         "researchforge": {
           "command": "/absolute/path/to/researchforge/scripts/mcp_wrapper.sh"
         }
       }
     }
     ```
   - SSE transport exposed on port 8001 via Docker port mapping for HTTP-based MCP clients

3. **Corpus feedback loop** in `src/researchforge/rag/feedback.py`
   - `maybe_ingest_briefing(briefing_id)` — scores briefing, ingests if above threshold
   - Heuristic scoring (pre-eval engine):
     - Word count > 200 → +0.2
     - Has citations → +0.2
     - Has multiple sections → +0.2
     - No error flags in pipeline trace → +0.2
     - Critic verdict was "pass" → +0.2
   - Threshold: 0.6 (configurable)
   - Chunks tagged with `content_type: "agent_generated"`, `briefing_id`, `quality_score`

4. **Expanded parsers** in `src/researchforge/rag/parsers.py`
   - `parse_html(path) -> str` — BeautifulSoup4 for local files, trafilatura for URLs
   - `parse_docx(path) -> str` — mammoth → Markdown
   - `parse_txt(path) -> str` — direct read

5. **Content type filtering** in retriever
   - `retrieve(query, source_only=True)` adds `WHERE content_type = 'source'` to LanceDB query

### What's Testable

- [x] MCP server starts and lists tools via stdio transport
- [x] `research` tool returns job_id and pipeline eventually completes
- [x] `query_corpus` tool returns relevant chunks
- [x] `ingest_document` tool processes a file and returns chunk count
- [x] `list_briefings` and `get_briefing` tools return correct data
- [x] `get_status` tool reports accurate pipeline progress
- [x] Briefing with score >= 0.6 is ingested into corpus
- [x] Briefing with score < 0.6 is NOT ingested into corpus
- [x] `source_only=True` excludes agent-generated chunks from search results
- [x] HTML and DOCX files parse correctly
- [x] Claude Desktop can connect and use all tools (manual test)

### Demo

```bash
# MCP via Claude Desktop uses the wrapper script
# (see scripts/mcp_wrapper.sh — runs docker compose exec under the hood)

# In Claude Desktop:
# > "Use researchforge to research the history of neural network architectures"
# > "Search the researchforge knowledge base for 'attention mechanism'"
# > "Ingest this PDF into researchforge: /app/data/sample_docs/transformer.pdf"

# Test MCP server manually
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | \
  docker compose exec -T app python -m researchforge mcp
```

### Testing Strategy

| Test | Type | Requires Ollama? |
|------|------|-------------------|
| MCP tool definitions (schema validation) | Unit | No |
| MCP tool execution (mock pipeline) | Integration | No |
| Corpus feedback scoring heuristics | Unit | No |
| Feedback loop integration | Integration | Yes |
| HTML / DOCX parsing | Unit | No |
| Claude Desktop integration | Manual E2E | Yes |

---

## Phase 5 — Eval Engine + Polish

**Goal:** Build the evaluation system for measuring and improving quality across the pipeline. Add model benchmarking. Harden the system for portfolio presentation.

**Complexity: L**

**Depends on: Phase 4**

### What's Built

| Component | Scope |
|-----------|-------|
| **Retrieval eval** | Precision/Recall/MRR against labeled test sets |
| **Agent eval** | Rubric-based LLM-as-judge scoring per agent |
| **End-to-end eval** | Briefing quality against reference briefings |
| **Model benchmarking** | Compare models per agent role |
| **Regression detection** | Alert when scores drop |
| **Eval dashboard** | Web UI page with charts and comparison tables |
| **Eval datasets** | Bootstrap set of test cases |
| **Human scoring UI** | Rate briefings in the web UI |
| **Polish** | Error handling hardening, startup checks, README, setup script |

### Specific Deliverables

1. **`src/researchforge/eval/retrieval_eval.py`**
   - `evaluate_retrieval(test_set_path) -> RetrievalEvalResult`
   - Computes Precision@K, Recall@K, MRR for each test case
   - Test set format: `{"question": "...", "relevant_chunk_ids": [...]}`

2. **`src/researchforge/eval/judge.py`**
   - `LLMJudge` class — calls Ollama with eval rubric prompt
   - Scores agent output on 5 criteria (structural validity, relevance, completeness, coherence, conciseness)
   - Uses `format: {json_schema}` for structured score output
   - Runs 3 times per evaluation, takes median score per criterion

3. **`src/researchforge/eval/agent_eval.py`**
   - `evaluate_agent(agent_name, test_cases) -> AgentEvalResult`
   - Feeds test inputs to an agent, scores output via LLMJudge
   - Special Critic test: plant errors, check detection rate

4. **`src/researchforge/eval/e2e_eval.py`**
   - `evaluate_e2e(reference_dir) -> E2EEvalResult`
   - Runs full pipeline on reference topics
   - Compares output briefings to reference briefings via LLMJudge
   - Scores: factual accuracy, coverage, attribution, readability

5. **`src/researchforge/eval/benchmark.py`**
   - `benchmark_models(agent_role, models: list[str]) -> BenchmarkResult`
   - Runs the same eval suite with different models for a given role
   - Records scores + latency + token counts per model
   - Output: comparison matrix (JSON)

6. **`src/researchforge/eval/runner.py`**
   - `run_eval_suite() -> FullEvalResult` — runs all eval dimensions
   - CLI: `python -m researchforge eval run`
   - CLI: `python -m researchforge eval benchmark --role analyst --models "qwen2.5:14b,phi4:14b"`
   - Results appended to `eval/results/*.jsonl`

7. **Regression detection**
   - After each eval run, compare scores to the rolling average of the last 5 runs
   - Flag any metric that drops >10% as a regression
   - Regressions logged and displayed in the eval dashboard

8. **Bootstrap eval datasets** in `eval/datasets/`
   - `retrieval_test.jsonl` — 20 question-answer pairs (manually created from seeded corpus)
   - `critic_test.jsonl` — 10 analyses with planted errors
   - `reference_briefings/` — 5 reference briefings on diverse topics

9. **Eval dashboard** — new route + template
   - `GET /eval` — eval dashboard HTML
   - Charts: retrieval quality over time, agent scores by role, model benchmark table
   - Charts rendered server-side as simple HTML tables/bars (no JS charting library needed — CSS bar charts)
   - Or: use lightweight `chart.css` or render SVG bar charts from Jinja2

10. **Human scoring in briefing viewer**
    - Thumbs up / thumbs down button on each briefing
    - Optional: click to expand detailed rubric (5 criteria, 1-5 scale each)
    - Stored in `briefings` table: `human_score`, `human_rubric_scores`

11. **Polish and hardening**
    - Startup health check: verify Ollama running, models pulled, data dirs exist
    - Missing model auto-suggestion: "Model X not found. Run: ollama pull X"
    - Graceful shutdown: cancel running pipelines on SIGINT
    - Comprehensive README.md with setup instructions, screenshots, architecture diagram
    - `scripts/setup_models.sh` — pull all required models
    - `scripts/seed_corpus.py` — seed corpus with sample documents for demo

### What's Testable

- [x] Retrieval eval computes correct Precision/Recall on a known test set
- [x] LLMJudge produces valid rubric scores (0.0-1.0 per criterion)
- [x] Agent eval correctly identifies strong vs. weak agent outputs
- [x] Critic eval detects planted errors (detection rate > 60% on test set)
- [x] E2E eval produces meaningful quality scores for reference topics
- [x] Model benchmark produces comparison matrix with scores and latency
- [x] Regression detection fires when scores drop >10%
- [x] Eval dashboard renders charts and tables correctly
- [x] Human scoring persists to database
- [x] `python -m researchforge eval run` completes without errors

### Demo

```bash
# Run the full eval suite
docker compose exec app python -m researchforge eval run
# → Retrieval: P@5=0.72, R@5=0.68, MRR=0.81
# → Agent scores: Planner=0.78, Gatherer=0.71, Analyst=0.74, Critic=0.69, Writer=0.82
# → E2E: Average briefing quality=0.73

# Benchmark models for the analyst role
docker compose exec app python -m researchforge eval benchmark \
  --role analyst --models "qwen2.5:14b,phi4:14b,qwen2.5:7b"
# → qwen2.5:14b: score=0.74, latency=45s (GPU-accelerated)
# → phi4:14b:    score=0.76, latency=52s
# → qwen2.5:7b:  score=0.61, latency=22s

# View eval dashboard (web UI already running)
# Open http://localhost:8000/eval
```

### Testing Strategy

| Test | Type | Requires Ollama? |
|------|------|-------------------|
| Eval metric computation (P/R/MRR) | Unit | No |
| LLMJudge output parsing | Unit | No (mock Ollama response) |
| LLMJudge scoring accuracy | Integration | Yes |
| Regression detection logic | Unit | No |
| Eval runner orchestration | Integration | Yes |
| Benchmark runner | Integration | Yes |
| Eval dashboard rendering | Unit | No |

---

## Phase Dependencies

```
Phase 1: Vertical Slice
    │
    ▼
Phase 2: Full Agent Pipeline
    │
    ▼
Phase 3: Web UI
    │
    ▼
Phase 4: MCP Server + Corpus Feedback
    │
    ▼
Phase 5: Eval Engine + Polish
```

Each phase strictly depends on the previous one. However, some work within phases can be parallelized:

- **Phase 3 + 4**: The MCP server (Phase 4) can be started in parallel with the Web UI (Phase 3) since they share the same backend APIs. If working with a collaborator, one person could do Web UI while the other does MCP.
- **Phase 5 eval datasets**: Can be created at any time (they're just JSONL files). Starting them during Phase 2 testing is recommended.

---

## Complexity Sizing Reference

| Size | Meaning | Example |
|------|---------|---------|
| **S** | < 1 day of focused work. Few files, clear path. | Add a new parser for a file format |
| **M** | 2-4 days of focused work. Multiple files, some design decisions. | Build the Web UI with SSE streaming |
| **L** | 1-2 weeks of focused work. Many files, complex interactions, integration testing. | Full 5-agent LangGraph pipeline with retries |
| **XL** | 2+ weeks. Cross-cutting concerns, significant debugging expected. | The entire project |

---

## Cross-Cutting Concerns (All Phases)

### Testing Convention

```python
# tests/conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "ollama: requires running Ollama instance")

@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal test PDF."""
    ...

@pytest.fixture
def seeded_corpus(tmp_path):
    """Return a VectorStore with pre-loaded test chunks (mock embeddings)."""
    ...

@pytest.fixture
def mock_ollama(httpx_mock):
    """Mock Ollama HTTP responses for unit tests."""
    ...
```

**Test running:**
```bash
# All tests except those requiring Ollama
docker compose exec app pytest -m "not ollama"

# All tests (requires Ollama container healthy)
docker compose exec app pytest

# Specific phase
docker compose exec app pytest tests/test_rag/
docker compose exec app pytest tests/test_agents/
```

### Git Strategy

- One branch per phase: `phase-1/vertical-slice`, `phase-2/agent-pipeline`, etc.
- Merge to `main` when phase is complete and tested
- Tag each phase completion: `v0.1.0` (Phase 1), `v0.2.0` (Phase 2), etc.

### Documentation Strategy

- Phase 1: README with setup instructions
- Phase 2: Add architecture diagram to README
- Phase 3: Add screenshots to README
- Phase 5: Comprehensive README with all sections, link to DESIGN.md

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NVIDIA Container Toolkit not installed on host | Medium | High | Detect at startup; provide clear install instructions; fall back to `docker-compose.cpu.yml` |
| GPU not detected inside Ollama container | Medium | Medium | Ollama auto-falls back to CPU; health check reports GPU status; log warning in UI |
| Ollama container OOM on large models | Medium | Medium | Docker memory limits; fall back to 7B models; document VRAM requirements per model |
| Docker volume permissions mismatch | Medium | Low | Dockerfile creates dirs with correct ownership; document `chown` fix for bind mounts |
| Small models produce garbage JSON | High | Medium | Use Ollama `format: {schema}` enforcement + structural retries |
| LangGraph API changes between versions | Medium | Medium | Pin exact version in pyproject.toml; lock in Dockerfile |
| LanceDB hybrid search underperforms | Low | Medium | Fall back to bm25s + manual RRF fusion (DESIGN.md Option C) |
| Ollama slow on CPU-only hardware | High | Low | Use 7B models; set expectations for 2-10 min per pipeline; document GPU recommendation |
| Critic agent too lenient (always passes) | Medium | Medium | Tune prompt to be explicitly adversarial; eval engine detects in Phase 5 |
| Critic agent too strict (never passes) | Medium | Medium | Max retry limit (2) + best-effort Writer fallback |
| Context window overflow with many chunks | Low | Medium | Budget system enforces max tokens per section (DESIGN.md §11.3) |
| pymupdf4llm produces poor Markdown from scanned PDFs | Medium | Low | Document that scanned PDFs need OCR; out of scope for v1 |
| Docker Compose version incompatibility | Low | Medium | Document minimum versions (Engine 24.0+, Compose 2.20+); test on clean install |

---

## Quick Start Checklist (for After Implementation)

```bash
# 1. Clone
git clone <repo-url> && cd researchforge

# 2. Start the stack (one command)
docker compose up -d
# → Builds the app image (first time)
# → Starts Ollama container with GPU passthrough
# → Starts app container with web UI on port 8000

# CPU-only host (no NVIDIA GPU):
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d

# 3. Pull models (first time only — ~30 GB total, takes a while)
docker compose exec ollama bash scripts/setup_models.sh

# 4. Verify GPU is being used
docker compose exec ollama nvidia-smi

# 5. Seed sample corpus
docker compose exec app python scripts/seed_corpus.py

# 6. Use it
open http://localhost:8000                                              # Web UI
docker compose exec app python -m researchforge research "..."          # CLI research
docker compose exec -T app python -m researchforge mcp                  # MCP server
docker compose exec app python -m researchforge eval run                # Run eval suite

# Useful commands
docker compose logs -f                   # Tail all logs
docker compose logs -f ollama            # Tail Ollama logs (see GPU usage)
docker compose exec app bash             # Shell into app container
docker compose down                      # Stop everything (data persists)
docker compose down -v                   # Stop + delete Ollama models volume
```
