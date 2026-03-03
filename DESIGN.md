# ResearchForge — Design Document

> A local-first, multi-agent research analyst platform powered by Ollama.
> Version: 0.1.0-draft | Last updated: 2026-03-03

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Multi-Agent Orchestration](#3-multi-agent-orchestration)
4. [RAG System](#4-rag-system)
5. [MCP Server](#5-mcp-server)
6. [Eval Engine](#6-eval-engine)
7. [User Interface](#7-user-interface)
8. [Tech Stack](#8-tech-stack)
9. [Configuration Schema](#9-configuration-schema)
10. [Error Handling Strategy](#10-error-handling-strategy)
11. [Agent Prompt Design Guidelines](#11-agent-prompt-design-guidelines)
12. [Project Directory Structure](#12-project-directory-structure)
13. [Docker & Containerization](#13-docker--containerization)

---

## 1. Project Overview

**ResearchForge** is a local multi-agent system where specialized AI agents collaborate to research, analyze, cross-reference, and produce structured briefings on any topic. It runs entirely on consumer hardware using Ollama models, builds a persistent RAG knowledge corpus over time, exposes itself as an MCP server, and uses an eval engine to measure and improve quality.

### Core Principles

- **100% local** — No paid APIs, no cloud GPU, no SaaS dependencies
- **Dockerized** — Every service runs in containers via Docker Compose; `docker compose up` is the only command needed to run the full stack
- **GPU-accelerated Ollama** — Ollama runs in its own container with NVIDIA GPU passthrough for 10-30x faster inference
- **Ollama as the sole LLM backend** — All inference goes through the Ollama container
- **Open source friendly** — Clone, `docker compose up`, done
- **Dual-use** — Standalone tool + MCP server for other AI tools to query

### Hardware Requirements

| Tier | RAM | GPU | Docker | Experience |
|------|-----|-----|--------|-----------|
| Minimum | 16 GB | None (CPU-only) | Docker Engine + Compose | Single 7B model, sequential agents, slower |
| Recommended | 32 GB | NVIDIA GPU (8+ GB VRAM) | Docker Engine + Compose + NVIDIA Container Toolkit | 14B models, GPU-accelerated, fast |
| Optimal | 32 GB | NVIDIA GPU (12+ GB VRAM) | Docker Engine + Compose + NVIDIA Container Toolkit | 14B models fully in VRAM, fastest |

### Prerequisites

| Software | Required | Purpose |
|----------|----------|---------|
| Docker Engine | Yes | Container runtime (v24.0+) |
| Docker Compose | Yes | Multi-container orchestration (v2.20+) |
| NVIDIA Driver | For GPU | Host GPU driver (v535+) |
| NVIDIA Container Toolkit | For GPU | Exposes GPU to Docker containers |

> **CPU-only fallback:** If no NVIDIA GPU is available, the Ollama container runs on CPU automatically. The `docker-compose.yml` includes a `docker-compose.cpu.yml` override that removes GPU configuration for CPU-only hosts.

---

## 2. Architecture Overview

### System Architecture Diagram

```
┌─── Docker Compose Network (researchforge) ───────────────────────────┐
│                                                                      │
│  ┌─── Container: app ────────────────────────────────────────────┐   │
│  │                                                                │   │
│  │  ┌─────────────┐    ┌──────────────────────────────────┐      │   │
│  │  │  Web UI      │    │  MCP Server (FastMCP)            │      │   │
│  │  │  (FastAPI +  │    │  Tools: research, query_corpus,  │      │   │
│  │  │   HTMX)      │    │  ingest_document, list_briefings,│      │   │
│  │  │  :8000 ──────┼────┼▶ Host port 8000                  │      │   │
│  │  └──────┬───────┘    └──────────┬───────────────────────┘      │   │
│  │         └────────┬──────────────┘                              │   │
│  │                  ▼                                              │   │
│  │  ┌──────────────────────────────────────────────────────┐      │   │
│  │  │          Pipeline Orchestrator (LangGraph)            │      │   │
│  │  │                                                      │      │   │
│  │  │  ┌────────┐  ┌─────────┐  ┌─────────┐              │      │   │
│  │  │  │Planner │─▶│Gatherer │─▶│Analyst  │              │      │   │
│  │  │  └────────┘  └─────────┘  └────┬────┘              │      │   │
│  │  │                                ▼                     │      │   │
│  │  │                       ┌──────────────┐               │      │   │
│  │  │                       │ Critic Agent │               │      │   │
│  │  │                       └──────┬───────┘               │      │   │
│  │  │                     ┌────────┴────────┐              │      │   │
│  │  │                     │Pass? Y→Writer   │              │      │   │
│  │  │                     │      N→Analyst  │              │      │   │
│  │  │                     └─────────────────┘              │      │   │
│  │  └──────────────────────────────────────────────────────┘      │   │
│  │                  │                    │                          │   │
│  │       ┌──────────┘        ┌───────────┘                        │   │
│  │       ▼                   ▼                                    │   │
│  │  ┌──────────────┐  ┌──────────────┐                            │   │
│  │  │ RAG System   │  │ Eval Engine  │                            │   │
│  │  │ LanceDB      │  │ Rubric       │                            │   │
│  │  │ BM25/FTS     │  │ scoring      │                            │   │
│  │  │ SQLite       │  │ Benchmarks   │                            │   │
│  │  └──────────────┘  └──────────────┘                            │   │
│  │         │                                                      │   │
│  │    Volumes: /app/data, /app/logs                               │   │
│  └────────────────────────────────────────────────────────────────┘   │
│         │                                                            │
│         │ HTTP (ollama:11434)                                        │
│         ▼                                                            │
│  ┌─── Container: ollama ─────────────────────┐                       │
│  │                                            │                       │
│  │  Ollama Server                             │                       │
│  │  LLMs + Embeddings                         │                       │
│  │  GPU: NVIDIA passthrough (deploy.resources) │                       │
│  │  Volume: ollama_models (persistent)        │                       │
│  │                                            │                       │
│  └────────────────────────────────────────────┘                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow — Full Research Pipeline

```
User submits research question
        │
        ▼
┌──────────────────┐
│  1. PLANNER      │  Input:  Research question
│                  │  Output: Structured research plan (JSON)
│  Model: deepseek │         - Sub-questions (3-7)
│  -r1:14b         │         - Information needs per sub-question
│                  │         - Suggested source types
│                  │         - Priority ordering
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  2. GATHERER     │  Input:  Research plan + RAG corpus access
│                  │  Output: Evidence collection (JSON)
│  Model: qwen2.5  │         - Retrieved chunks per sub-question
│  :7b             │         - Source attributions
│                  │         - Relevance assessments
│                  │         - Gaps identified
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  3. ANALYST      │  Input:  Evidence collection + original question
│                  │  Output: Analysis document (JSON)
│  Model: qwen2.5  │         - Key findings per sub-question
│  :14b            │         - Cross-references between sources
│                  │         - Contradictions flagged
│                  │         - Confidence levels per finding
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  4. CRITIC       │  Input:  Analysis document + evidence
│                  │  Output: Review verdict (JSON)
│  Model: deepseek │         - Pass/Revise decision
│  -r1:7b          │         - List of issues (if any)
│                  │         - Missing perspectives
│                  │         - Unsupported claims flagged
│                  │         - Suggested improvements
└────────┬─────────┘
         │
    ┌────┴────┐
    │ Pass?   │
    │         │
   Yes       No ──▶ Back to Analyst (max 2 retries)
    │
    ▼
┌──────────────────┐
│  5. WRITER       │  Input:  Reviewed analysis + evidence
│                  │  Output: Final briefing (Markdown)
│  Model: mistral  │         - Executive summary
│  -nemo:12b       │         - Detailed findings (sections)
│                  │         - Source citations
│                  │         - Confidence assessment
│                  │         - Further research suggestions
└────────┬─────────┘
         │
         ▼
    Final briefing stored in SQLite +
    optionally ingested into RAG corpus
```

---

## 3. Multi-Agent Orchestration

### 3.1 Communication Pattern: Orchestrator-Driven with Shared State

**Decision: Orchestrator-driven pipeline using LangGraph's StateGraph.**

The orchestrator owns a single typed state dict that flows through the graph. Each agent node reads from state, does its work, and writes results back to state. There is no direct agent-to-agent messaging.

**Rationale:**
- **Not message-passing**: Small models (7B-14B) struggle with multi-turn conversational negotiation. They work best with a single, focused prompt per invocation.
- **Not pure blackboard**: A blackboard allows any agent to read/write anything at any time, which makes the flow hard to trace. The orchestrator enforces a defined execution order.
- **Orchestrator-driven**: Gives explicit control over transitions, retries, and parallelism. The execution graph is visible and debuggable.

**State Schema:**

```python
from typing import TypedDict, Literal
from dataclasses import dataclass, field

class PipelineState(TypedDict):
    # Input
    research_question: str

    # Planner output
    research_plan: dict | None           # Sub-questions, info needs, priorities

    # Gatherer output
    evidence: list[dict] | None          # Retrieved chunks with attributions
    gaps: list[str] | None               # Info gaps identified

    # Analyst output
    analysis: dict | None                # Findings, cross-refs, contradictions

    # Critic output
    critic_verdict: str | None           # "pass" or "revise"
    critic_issues: list[str] | None      # Issues found
    revision_count: int                  # Track retry attempts

    # Writer output
    briefing: str | None                 # Final markdown briefing

    # Pipeline metadata
    pipeline_id: str
    status: str                          # "planning", "gathering", "analyzing", etc.
    errors: list[str]
    trace: list[dict]                    # Timestamped log of all agent actions
```

### 3.2 Disagreement Handling: Critic-Analyst Retry Loop

When the Critic rejects the Analyst's work:

```
Analyst produces analysis
        │
        ▼
  Critic reviews ──▶ "pass" ──▶ Writer
        │
      "revise"
        │
        ▼
  revision_count += 1
        │
   ┌────┴────┐
   │ count   │
   │ <= 2?   │
   │         │
  Yes       No ──▶ Writer proceeds with best-effort analysis
   │                (critic issues appended as caveats in briefing)
   ▼
  Analyst re-runs with:
    - Original evidence
    - Critic's specific issues as additional instructions
    - Previous analysis (to avoid regressing)
```

**Max retries: 2.** After 2 failed critic reviews, the pipeline continues to the Writer, but the briefing includes a "Caveats" section listing unresolved critic concerns. This prevents infinite loops with small models that may never fully satisfy the critic.

**Implementation in LangGraph:**

```python
def should_revise(state: PipelineState) -> Literal["analyst", "writer"]:
    if state["critic_verdict"] == "pass":
        return "writer"
    if state["revision_count"] >= 2:
        return "writer"  # Best-effort with caveats
    return "analyst"

graph.add_conditional_edges("critic", should_revise)
```

### 3.3 Low-Quality Output Fallback Strategy

Small models will sometimes produce malformed, off-topic, or low-quality output. The fallback strategy has three tiers:

**Tier 1 — Structural Retry (automatic)**
If an agent's output fails JSON schema validation or is empty:
- Retry the same prompt up to 2 times with the same model
- On retry, append to the prompt: `"Your previous response was invalid. You MUST respond with valid JSON matching this schema: {schema}"`
- Use Ollama's `format: {json_schema}` to enforce structure

**Tier 2 — Model Downgrade (automatic, for specific failure patterns)**
If a 14B model times out (>120s) or produces out-of-memory errors:
- Automatically fall back to the configured 7B alternative for that role
- Log the fallback in `state["trace"]`
- Example: `qwen2.5:14b` times out → fall back to `qwen2.5:7b`

**Tier 3 — Graceful Degradation (pipeline continues)**
If an agent fails after all retries:
- The pipeline continues with a placeholder result
- Downstream agents receive a flag indicating upstream failure
- The final briefing includes a "Data Quality" section noting which stages had issues
- The eval engine records the failure for regression tracking

**Configuration:**

```yaml
agents:
  analyst:
    primary_model: "qwen2.5:14b"
    fallback_model: "qwen2.5:7b"
    max_retries: 2
    timeout_seconds: 120
```

### 3.4 Sequential vs. Parallel Execution

**Default: Sequential execution.** The pipeline is inherently sequential — each agent depends on the previous agent's output.

**One exception — Gatherer parallelism:**
The Gatherer can execute sub-question retrievals in parallel, since each RAG query is independent:

```
Research Plan has sub-questions [Q1, Q2, Q3, Q4]
        │
        ▼
   ┌────┴────┬────┬────┐
   ▼         ▼    ▼    ▼
 RAG(Q1)  RAG(Q2) RAG(Q3) RAG(Q4)   ← asyncio.gather()
   │         │    │    │
   └────┬────┴────┴────┘
        ▼
  Combined evidence collection
```

The RAG queries themselves are fast (vector search + BM25), so this parallelism speeds up the Gatherer step. The LLM calls (summarizing retrieved chunks) remain sequential because Ollama serves one inference at a time on consumer hardware.

**Why not parallel agents?**
On a single machine with 16-32GB RAM, Ollama can only run one LLM inference at a time (unless you have enough VRAM/RAM for two loaded models). Parallel agent LLM calls would just queue behind each other. Sequential execution is simpler and equally fast given the hardware constraint.

### 3.5 Pipeline State Management and Observability

**Logging:**
- Every agent invocation is logged to `state["trace"]` as a timestamped entry:
  ```json
  {
    "timestamp": "2026-03-03T10:15:32Z",
    "agent": "analyst",
    "model": "qwen2.5:14b",
    "input_tokens": 2847,
    "output_tokens": 1523,
    "duration_ms": 45200,
    "status": "success",
    "fallback_used": false
  }
  ```

**Tracing:**
- LangGraph's built-in `SqliteSaver` checkpoints the full state after each node execution
- This enables: resuming failed pipelines, inspecting intermediate state, replaying specific steps
- Checkpoints stored in `data/checkpoints.db`

**Real-time progress:**
- The Web UI and MCP server receive SSE events as each agent starts/completes
- Events include: agent name, status, duration, summary of output

**Structured logging to file:**
- Python `structlog` → JSON lines to `logs/pipeline.jsonl`
- Includes pipeline_id for correlating all events in a single research run

### 3.6 Recommended Ollama Models per Agent Role

#### 32 GB RAM Configuration (Recommended)

| Agent Role | Model | Ollama Tag | RAM (loaded) | Rationale |
|------------|-------|------------|-------------|-----------|
| Planner | DeepSeek-R1 14B | `deepseek-r1:14b` | ~9.5 GB | Explicit chain-of-thought via `<think>` blocks; best at decomposing complex questions into structured sub-plans |
| Gatherer | Qwen 2.5 7B | `qwen2.5:7b` | ~4.8 GB | Fast, reliable instruction following; good at structured output for query formulation; lower RAM allows quick load/unload |
| Analyst | Qwen 2.5 14B | `qwen2.5:14b` | ~9.5 GB | Strong analytical reasoning; 128K context window; excellent at synthesizing multiple sources |
| Critic | DeepSeek-R1 7B | `deepseek-r1:7b` | ~4.8 GB | Reasoning traces make critique chains visible and debuggable; naturally adversarial |
| Writer | Mistral-Nemo 12B | `mistral-nemo:12b` | ~7.5 GB | Best prose quality in the 7B-14B range; fluent, structured writing |
| Embeddings | nomic-embed-text | `nomic-embed-text` | ~0.3 GB | 768-dim, Matryoshka support, task prefixes (`search_query:` / `search_document:`), tiny footprint |

**Memory note:** Only one LLM model is loaded at a time. Ollama unloads the previous model before loading the next (configurable via `OLLAMA_KEEP_ALIVE`). The embedding model stays resident at all times (0.3 GB). Peak usage: ~10 GB (14B model + embedding + OS).

**Docker note:** The app container communicates with Ollama via the Docker Compose network using the service name `ollama` (i.e., `http://ollama:11434`), not `localhost`. The Ollama container has GPU access via `deploy.resources.reservations.devices` in `docker-compose.yml`.

#### 16 GB RAM Configuration (Minimum)

| Agent Role | Model | Ollama Tag | RAM |
|------------|-------|------------|-----|
| All LLM roles | Qwen 2.5 7B | `qwen2.5:7b` | ~4.8 GB |
| Embeddings | nomic-embed-text | `nomic-embed-text` | ~0.3 GB |

On 16 GB: use a single model for all roles, differentiated only by system prompt. Set `OLLAMA_KEEP_ALIVE=0` to unload immediately after each call to free RAM for the OS.

---

## 4. RAG System

### 4.1 Document Ingestion Pipeline

```
Input Document (PDF/MD/HTML/DOCX/TXT)
        │
        ▼
┌──────────────────┐
│  1. PARSE        │  Extract text + preserve structure
│                  │  PDF  → pymupdf4llm (outputs Markdown)
│                  │  HTML → trafilatura (web) / BeautifulSoup4 (local)
│                  │  DOCX → mammoth (outputs Markdown)
│                  │  MD   → pass-through
│                  │  TXT  → pass-through
└────────┬─────────┘
         │  (Markdown text)
         ▼
┌──────────────────┐
│  2. CHUNK        │  Split into retrievable segments
│                  │  Strategy: RecursiveCharacterTextSplitter
│                  │  Chunk size: 1500 chars (~375 tokens)
│                  │  Overlap: 200 chars (~50 tokens)
│                  │  Separators: ["\n\n", "\n", ". ", " "]
│                  │
│                  │  For structured docs (Markdown output):
│                  │  First pass: MarkdownHeaderTextSplitter
│                  │  Second pass: RecursiveCharacterTextSplitter
│                  │  (preserves section headers as metadata)
└────────┬─────────┘
         │  (chunks with metadata)
         ▼
┌──────────────────┐
│  3. EMBED        │  Generate vector embeddings
│                  │  Model: nomic-embed-text (via Ollama)
│                  │  Dimensions: 768
│                  │  Prefix: "search_document: " prepended to each chunk
│                  │  Batch size: 32 chunks per Ollama call
└────────┬─────────┘
         │  (chunks + vectors + metadata)
         ▼
┌──────────────────┐
│  4. STORE        │  Persist to LanceDB + build FTS index
│                  │  Vector table: chunks + embeddings
│                  │  FTS index: Tantivy full-text on chunk text
│                  │  Metadata table: SQLite (source tracking)
└──────────────────┘
```

### 4.2 Parser Library Choices

| Format | Library | Package | Why This One |
|--------|---------|---------|-------------|
| PDF | pymupdf4llm | `pymupdf4llm` | Outputs clean Markdown preserving headers, lists, tables. Built on PyMuPDF (fast). Purpose-built for RAG pipelines. |
| HTML (web) | trafilatura | `trafilatura` | Removes boilerplate (nav, footer, ads), extracts article body + metadata (author, date). Best for web content. |
| HTML (local) | BeautifulSoup4 | `beautifulsoup4` | Full control for known-structure local HTML files. |
| DOCX | mammoth | `mammoth` | Converts to Markdown preserving semantic structure (headings → `#`). Clean output for chunking. |
| Markdown | (direct) | — | Already in target format. Pass to chunker directly. |
| TXT | (direct) | — | Read and pass to chunker. |

### 4.3 Chunking Strategy

**Primary strategy: Two-pass chunking for structured documents.**

**Pass 1 — Structure-aware split (Markdown documents):**
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = md_splitter.split_text(markdown_text)
# Each chunk has metadata: {"h1": "Introduction", "h2": "Methods"}
```

**Pass 2 — Size-constrained split:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,        # ~375 tokens (safe for 7B-14B models)
    chunk_overlap=200,      # ~50 tokens overlap
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)
final_chunks = text_splitter.split_documents(md_chunks)
```

**For plain text (no structure):**
Skip Pass 1 — use `RecursiveCharacterTextSplitter` directly.

**Why these numbers:**
- **1500 chars (~375 tokens)**: With 3-5 chunks retrieved per query, total injected context is ~1125-1875 tokens. This fits comfortably in a 7B model's effective attention span while leaving room for the system prompt and generation.
- **200 chars overlap**: ~13% overlap prevents losing context at chunk boundaries. Sentences split across chunks are captured in both.
- **RecursiveCharacterTextSplitter**: Tries paragraph breaks first, then line breaks, then sentences, then words. This produces semantically coherent chunks most of the time without the overhead of semantic chunking (which requires embedding each sentence during indexing).

### 4.4 Vector Storage — LanceDB

**Decision: LanceDB** as the vector store.

**Rationale:**
| Criterion | LanceDB | ChromaDB | Qdrant |
|-----------|---------|----------|--------|
| Embedded (no server) | By design | Yes | Via separate wheel |
| Built-in hybrid search | Yes (Tantivy FTS) | No | Yes (sparse vectors) |
| Storage format | Open (Apache Lance) | Internal | Internal (RocksDB) |
| Python API feel | Pandas-native | Simple dict-based | Verbose/type-heavy |
| Data portability | Excellent (Lance = open columnar) | Moderate | Moderate |
| Dependencies | pyarrow, lance | sqlite3, hnswlib | Rust wheel |

**LanceDB Schema:**

```python
import lancedb
import pyarrow as pa

CHUNKS_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),          # UUID
    pa.field("text", pa.string()),              # Chunk text content
    pa.field("vector", pa.list_(pa.float32(), 768)),  # nomic-embed-text embedding
    pa.field("document_id", pa.string()),       # FK to documents table
    pa.field("source_path", pa.string()),       # Original file path
    pa.field("source_type", pa.string()),       # "pdf", "html", "markdown", etc.
    pa.field("content_type", pa.string()),      # "source" or "agent_generated"
    pa.field("section_h1", pa.string()),        # From MarkdownHeaderTextSplitter
    pa.field("section_h2", pa.string()),        # From MarkdownHeaderTextSplitter
    pa.field("chunk_index", pa.int32()),        # Position within document
    pa.field("ingested_at", pa.string()),       # ISO timestamp
    pa.field("briefing_id", pa.string()),       # If agent-generated, which briefing
    pa.field("quality_score", pa.float32()),     # Eval score (null for source docs)
])
```

### 4.5 Embedding Model — nomic-embed-text via Ollama

**Model:** `nomic-embed-text` (v1.5)
**Dimensions:** 768 (supports Matryoshka truncation to 64/128/256/512)
**Size on disk:** ~274 MB
**RAM when loaded:** ~300 MB

**Usage pattern:**
```python
# Embedding at ingestion time
prefix = "search_document: "
embedding = ollama.embeddings(model="nomic-embed-text", prompt=prefix + chunk_text)

# Embedding at query time
prefix = "search_query: "
embedding = ollama.embeddings(model="nomic-embed-text", prompt=prefix + query_text)
```

The `search_document:` / `search_query:` prefixes are required by nomic-embed-text and improve retrieval quality by ~5-10% compared to unprefixed embeddings.

### 4.6 Retrieval Strategy — Hybrid Search with Reranking

```
User query
    │
    ├──▶ Embed query (nomic-embed-text, "search_query:" prefix)
    │         │
    │         ▼
    │    Vector search (top 20)
    │
    ├──▶ BM25/FTS search (top 20)
    │
    └──▶ Metadata filter (optional: source_type, date range, content_type)
              │
              ▼
       Reciprocal Rank Fusion (RRF)
       k=60, combine vector + BM25 rankings
              │
              ▼
       Top 5 chunks returned
       (with source attribution metadata)
```

**Implementation with LanceDB's built-in hybrid search:**

```python
# Build FTS index once per table
table.create_fts_index("text", replace=True)

# Hybrid query
results = (
    table.search(query_type="hybrid")
         .vector(query_embedding)          # Vector component
         .text(query_text)                 # BM25 component
         .where(f"content_type = 'source'")  # Optional: exclude agent-generated
         .limit(5)
         .rerank(reranker="rrf")           # Reciprocal Rank Fusion
         .to_list()
)
```

**Why RRF over score normalization:** BM25 scores and cosine similarity are on incompatible scales. RRF uses only rank positions, making it robust without requiring score calibration.

**Retrieval defaults:**
- Vector candidates: 20
- BM25 candidates: 20
- Final results after fusion: 5
- These are configurable per query

### 4.7 Corpus Growth — Feedback Loop Design

Every completed research briefing can be ingested back into the RAG corpus, so the system accumulates knowledge over time. But we must prevent low-quality outputs from polluting the corpus.

**Ingestion gate:**

```
Briefing completed
       │
       ▼
  Eval Engine scores briefing (0.0 - 1.0)
       │
  ┌────┴────┐
  │ Score   │
  │ >= 0.6? │
  │         │
  Yes      No ──▶ Stored in SQLite but NOT ingested into RAG corpus
   │                (available for review, not for retrieval)
   ▼
  Chunked and ingested with metadata:
    content_type: "agent_generated"
    briefing_id: "<uuid>"
    quality_score: 0.78
    ingested_at: "2026-03-03T..."
```

**Separation in retrieval:**
- By default, retrieval mixes source material and agent-generated content
- A `content_type` filter allows excluding agent-generated content:
  - `content_type = 'source'` — only original documents
  - `content_type = 'agent_generated'` — only previous briefings
  - No filter — both (default)
- The Gatherer agent's prompt instructs it to prefer source material and flag when a finding comes from a previous briefing vs. an original source

**Staleness:** Agent-generated content has an `ingested_at` timestamp. A future enhancement could weight recency or allow expiration of old agent-generated content.

### 4.8 Metadata and Provenance

**SQLite metadata schema** (`data/metadata.db`):

```sql
-- Source documents
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL,       -- "pdf", "html", "markdown", "docx", "txt"
    title TEXT,
    file_hash TEXT NOT NULL,         -- SHA-256 for dedup
    file_size_bytes INTEGER,
    ingested_at TEXT NOT NULL,
    chunk_count INTEGER
);

-- Research briefings
CREATE TABLE briefings (
    briefing_id TEXT PRIMARY KEY,
    research_question TEXT NOT NULL,
    status TEXT NOT NULL,            -- "running", "completed", "failed"
    started_at TEXT NOT NULL,
    completed_at TEXT,
    quality_score REAL,              -- Eval score (0.0 - 1.0)
    ingested_into_corpus INTEGER DEFAULT 0,  -- Boolean
    briefing_markdown TEXT,          -- Full briefing content
    pipeline_trace TEXT              -- JSON: full agent trace
);

-- Chunk-level lineage
CREATE TABLE chunk_lineage (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT,                -- FK to documents (null if agent-generated)
    briefing_id TEXT,                -- FK to briefings (null if source doc)
    content_type TEXT NOT NULL,      -- "source" or "agent_generated"
    chunk_index INTEGER,
    char_start INTEGER,              -- Position in original document
    char_end INTEGER
);
```

---

## 5. MCP Server

### 5.1 Tool Definitions

The MCP server exposes ResearchForge's capabilities to external AI tools (e.g., Claude Desktop, other MCP clients).

```python
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("researchforge")

@mcp.tool()
async def research(topic: str, depth: str = "standard") -> str:
    """Kick off a full multi-agent research pipeline on a topic.

    Args:
        topic: The research question or topic to investigate.
        depth: "quick" (planner+gatherer+writer) or "standard" (full pipeline with critic).

    Returns:
        A job ID for tracking progress. Use get_status() to poll.
    """

@mcp.tool()
async def query_corpus(query: str, limit: int = 5, source_only: bool = False) -> str:
    """Ask a question against the accumulated knowledge base using hybrid search.

    Args:
        query: Natural language question.
        limit: Number of results to return (1-20).
        source_only: If true, exclude agent-generated content from results.

    Returns:
        Relevant text chunks with source attributions.
    """

@mcp.tool()
async def ingest_document(file_path: str) -> str:
    """Add a new document to the RAG corpus.

    Args:
        file_path: Absolute path to a PDF, Markdown, HTML, DOCX, or TXT file.

    Returns:
        Document ID and chunk count.
    """

@mcp.tool()
async def list_briefings(limit: int = 10, status: str = "completed") -> str:
    """List completed research briefings.

    Args:
        limit: Number of briefings to return.
        status: Filter by status ("completed", "running", "failed", "all").

    Returns:
        List of briefings with IDs, topics, dates, and quality scores.
    """

@mcp.tool()
async def get_briefing(briefing_id: str) -> str:
    """Retrieve a specific research briefing by ID.

    Args:
        briefing_id: The UUID of the briefing.

    Returns:
        Full briefing content in Markdown with source citations.
    """

@mcp.tool()
async def get_status(job_id: str) -> str:
    """Check the status of a running research pipeline.

    Args:
        job_id: The job ID returned by the research() tool.

    Returns:
        Current pipeline stage, progress percentage, and any errors.
    """
```

### 5.2 Long-Running Pipeline Handling

Research pipelines take 2-15 minutes depending on model speed and question complexity. MCP tools should return quickly.

**Pattern: Async job submission with status polling.**

```
MCP Client calls research("impact of AI on healthcare")
        │
        ▼
  research() tool:
    1. Creates job entry in SQLite (status: "queued")
    2. Launches pipeline via asyncio.create_task()
    3. Returns immediately: {"job_id": "abc-123", "status": "queued"}
        │
        ▼
  MCP Client polls get_status("abc-123") periodically
        │
        ▼
  get_status() returns:
    {"job_id": "abc-123", "status": "analyzing", "stage": "analyst",
     "progress": 60, "elapsed_seconds": 180}
        │
        ▼
  Eventually:
    {"job_id": "abc-123", "status": "completed", "briefing_id": "def-456"}
        │
        ▼
  MCP Client calls get_briefing("def-456") to retrieve results
```

**Progress reporting via MCP Context:**
For MCP clients that support progress notifications:
```python
@mcp.tool()
async def research(topic: str, depth: str = "standard", ctx: Context = None) -> str:
    job_id = create_job(topic)
    task = asyncio.create_task(run_pipeline(job_id, topic, depth, ctx))
    return f"Job started: {job_id}"

async def run_pipeline(job_id, topic, depth, ctx):
    stages = ["planning", "gathering", "analyzing", "critiquing", "writing"]
    for i, stage in enumerate(stages):
        if ctx:
            await ctx.report_progress(progress=i, total=len(stages))
        await execute_stage(job_id, stage, ...)
```

### 5.3 Resource Management — Concurrent Pipelines

**Problem:** What if 3 research pipelines are triggered simultaneously? Ollama can only serve one inference at a time on consumer hardware.

**Solution: Pipeline queue with configurable concurrency.**

```python
import asyncio

class PipelineQueue:
    def __init__(self, max_concurrent: int = 1):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue: asyncio.Queue = asyncio.Queue()

    async def submit(self, job_id: str, topic: str, depth: str):
        """Submit a pipeline job. Returns immediately."""
        await self.queue.put((job_id, topic, depth))
        asyncio.create_task(self._process())

    async def _process(self):
        job_id, topic, depth = await self.queue.get()
        async with self.semaphore:
            await run_pipeline(job_id, topic, depth)
```

**Default `max_concurrent=1`**: On consumer hardware, running two pipelines simultaneously just doubles the time for both (Ollama queues inference requests internally). Sequential is more predictable.

**Status visibility:** Queued jobs show `status: "queued"` with their position in the queue.

### 5.4 Authentication / Access Control

**Decision: No authentication for the local MCP server.**

**Rationale:**
- The MCP server binds to `localhost` only — not accessible from the network
- MCP's stdio transport (the default for Claude Desktop integration) doesn't go over the network at all
- Adding auth adds complexity with no security benefit for a single-user local tool
- If someone later wants network access, they can add a reverse proxy with auth in front

**Configuration guard:**
```yaml
mcp:
  transport: "stdio"        # or "sse"
  bind_host: "127.0.0.1"   # Never "0.0.0.0"
  bind_port: 8001          # Only used for SSE transport
```

**Docker note on MCP:** The MCP server's stdio transport works by running the container command directly (e.g., `docker compose exec app python -m researchforge mcp`). For Claude Desktop integration, the MCP config points to a wrapper script that `docker compose exec`'s into the running container. The SSE transport binds to `0.0.0.0` inside the container but is only exposed to the host via Docker port mapping.

---

## 6. Eval Engine

### 6.1 Evaluation Dimensions

The eval engine measures quality across four dimensions:

#### A. Retrieval Quality

**What:** Does the RAG system return relevant chunks for a given query?

**Metrics:**
- **Precision@K**: Of the K chunks returned, how many are relevant?
- **Recall@K**: Of all relevant chunks in the corpus, how many were returned?
- **MRR (Mean Reciprocal Rank)**: How high is the first relevant result?

**Test sets:**
- Manually curated: 20-50 question-answer pairs where the answer exists in the corpus and the relevant chunks are labeled
- Format: `{"question": "...", "relevant_chunk_ids": ["id1", "id2"], "answer": "..."}`
- Stored in `eval/datasets/retrieval_test.jsonl`

**Scoring:** Automated — compare returned chunk IDs against labeled relevant chunk IDs.

#### B. Agent Output Quality

**What:** Does each agent produce useful, well-structured output?

**Metrics (rubric-based, scored 0.0-1.0):**

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Structural validity | 0.15 | Output matches expected JSON schema |
| Relevance | 0.30 | Content addresses the assigned task |
| Completeness | 0.25 | All requested components present |
| Coherence | 0.20 | Logical flow, no contradictions |
| Conciseness | 0.10 | No unnecessary padding or repetition |

**Scoring method:** LLM-as-judge using the same Ollama model (but a different instance/prompt than the agent being evaluated). The eval prompt provides the rubric, the agent's input, and the agent's output, and asks for a JSON score per criterion.

**Critic effectiveness test:**
- Plant known errors in an analysis (wrong dates, unsupported claims, logical fallacies)
- Feed to the Critic agent
- Score: did the Critic catch the planted errors?
- Stored in `eval/datasets/critic_test.jsonl`

#### C. End-to-End Quality

**What:** Is the final briefing good?

**Metrics:**
- **Factual accuracy**: Claims in the briefing are supported by source material
- **Coverage**: Key aspects of the topic are addressed
- **Attribution**: Sources are cited correctly
- **Readability**: Clear, well-organized prose

**Test sets:**
- 5-10 reference briefings on known topics, manually written or curated from high-quality sources
- The system runs the same topics and the output is compared against reference briefings
- Comparison via LLM-as-judge with a detailed rubric

#### D. Model Benchmarking

**What:** How do different Ollama models perform in each agent role?

**Method:**
- Run the same eval test sets with different models assigned to each role
- Record scores + latency + token usage per model
- Example comparison: `qwen2.5:14b` vs `phi4:14b` as Analyst

**Output:** A comparison matrix stored in `eval/results/model_benchmark.json`

### 6.2 Eval Scoring Implementation

**Primary: LLM-as-judge.** A dedicated eval model scores agent outputs against rubrics.

**Eval model recommendation:** Use the same model pool as the agents (e.g., `qwen2.5:14b`), but with a specialized eval system prompt. This keeps the system 100% local with no additional dependencies.

**Known limitation:** LLM-as-judge with small models is less reliable than GPT-4/Claude-based evaluation. Mitigations:
- Use structured output (JSON schema enforcement) for eval scores to prevent hallucinated scores
- Run each eval 3 times and take the median (reduces variance)
- Include heuristic checks alongside LLM scoring:
  - JSON schema validation (structural quality)
  - Word count thresholds (completeness proxy)
  - Source citation count (attribution proxy)
  - Flesch reading ease (readability proxy)

**Optional: Human-in-the-loop.** The Web UI allows manual scoring of briefings (thumbs up/down + optional detailed rubric). These human scores calibrate the LLM-as-judge over time.

### 6.3 Eval Dataset Creation and Maintenance

**Bootstrap strategy:**
1. Start with 5 manually written question-answer pairs and reference briefings
2. As you use the system, rate briefings via the UI (human scores)
3. High-scoring briefings (human-rated >= 0.8) become reference briefings for future evals
4. The eval dataset grows organically with usage

**Storage:**
```
eval/
  datasets/
    retrieval_test.jsonl       # Question → relevant chunks
    critic_test.jsonl          # Analysis with planted errors
    reference_briefings/       # Known-good briefings
      topic_1.md
      topic_2.md
  results/
    retrieval_scores.jsonl     # Timestamped eval results
    agent_scores.jsonl
    e2e_scores.jsonl
    model_benchmark.json       # Model comparison matrix
```

### 6.4 Eval Results Storage and Visualization

**Storage:** All eval results in `eval/results/` as JSON Lines (one result per line, timestamped). This enables:
- Append-only logging (no corruption risk)
- Easy filtering by date, model, agent role
- Simple ingestion into charts

**Visualization:** A dedicated page in the Web UI with:
- **Retrieval quality over time** — Line chart of Precision@5 / Recall@5 per eval run
- **Agent scores by role** — Bar chart comparing agents against rubric criteria
- **Model benchmark table** — Grid of models × roles with scores and latency
- **Regression alerts** — Highlighted when latest score drops >10% from the running average

---

## 7. User Interface

### 7.1 Tech Choice: FastAPI + HTMX + SSE

**Decision: FastAPI backend + HTMX frontend with SSE for real-time updates.**

**Rationale:**
- **FastAPI**: Async-native, excellent for SSE streaming, OpenAPI docs for free, well-known in the Python ecosystem
- **HTMX**: Minimal JavaScript, server-rendered HTML with dynamic updates. Attributes like `hx-get`, `hx-post`, `hx-ext="sse"` handle 90% of UI interactions
- **SSE (Server-Sent Events)**: Natural fit for streaming agent progress (token-by-token output, stage transitions). Simpler than WebSockets for one-directional server→client updates
- **Jinja2 templates**: Server-side rendering keeps the frontend simple and eliminates the need for a JS build step
- **No SPA framework needed**: Avoids React/Svelte build tooling complexity while still providing a responsive, modern-feeling UI

**Alternatives considered:**
- Gradio: Fastest path to a chat UI, but too opinionated for a dashboard with multiple views
- Streamlit: Execution model (full script re-run) fights async agent streaming
- React/Svelte: Overkill for this project's UI needs; adds a JS build step

### 7.2 UI Pages

**1. Research Dashboard (Home)**
- Text input for research question
- Depth selector: "Quick" / "Standard"
- "Start Research" button
- Active pipeline cards showing real-time agent progress via SSE:
  ```
  ┌─────────────────────────────────────────────┐
  │ "Impact of AI on healthcare"                │
  │ Status: ██████████░░░░░ Analyzing (60%)     │
  │ Planner ✓  Gatherer ✓  Analyst ⟳  Critic ·  Writer · │
  │ Elapsed: 3m 12s                             │
  └─────────────────────────────────────────────┘
  ```

**2. Briefing Viewer**
- List of completed briefings with search/filter
- Full briefing rendered as Markdown with:
  - Inline source citations (clickable, link to chunk viewer)
  - Quality score badge
  - Pipeline trace (expandable: agent timings, models used, retries)

**3. Knowledge Corpus Browser**
- Search the corpus (hybrid search box)
- Browse by source document
- View individual chunks with metadata
- Upload new documents (drag-and-drop)
- Corpus statistics: total documents, total chunks, storage size

**4. Eval Dashboard**
- Retrieval quality chart (Precision/Recall over time)
- Agent score breakdown (per role, per criterion)
- Model benchmark comparison table
- Run eval suite button
- Recent eval results with regression alerts

**5. Configuration Panel**
- Model assignment per agent role (dropdown of available Ollama models)
- Chunking parameters (chunk size, overlap)
- Retrieval parameters (top-K, vector weight vs BM25 weight)
- Pipeline settings (max retries, timeout, quality threshold for corpus ingestion)
- Ollama connection settings (host, port)

### 7.3 Real-Time Agent Progress via SSE

```python
# Backend: FastAPI SSE endpoint
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

@app.get("/api/pipelines/{job_id}/stream")
async def stream_pipeline(job_id: str):
    async def event_generator():
        async for event in pipeline_events(job_id):
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

```html
<!-- Frontend: HTMX SSE listener -->
<div hx-ext="sse"
     sse-connect="/api/pipelines/{{ job_id }}/stream"
     sse-swap="stage_update">
    <div id="progress">Waiting...</div>
</div>
```

**Event types:**
- `stage_start` — An agent has started (name, model)
- `stage_complete` — An agent has finished (name, duration, summary)
- `token` — Streaming token output from the Writer agent
- `error` — An agent encountered an error
- `pipeline_complete` — Full pipeline finished (briefing_id, quality_score)

---

## 8. Tech Stack

### 8.1 Complete Dependency List

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| **Infrastructure** | Docker Engine | >=24.0 | Container runtime |
| | Docker Compose | >=2.20 | Multi-container orchestration |
| | NVIDIA Container Toolkit | >=1.14 | GPU passthrough to Ollama container |
| | Ollama (container image) | latest | `ollama/ollama:latest` from Docker Hub |
| **Agent orchestration** | langgraph | >=0.2.0 | Graph-based agent pipeline |
| | langchain-ollama | >=0.2.0 | Ollama model provider for LangChain |
| | langchain-text-splitters | >=0.2.0 | Document chunking |
| | langchain-core | >=0.3.0 | Base abstractions |
| **Vector store** | lancedb | >=0.10.0 | Embedded vector DB with hybrid search |
| | pyarrow | >=14.0 | Arrow tables (LanceDB dependency) |
| **Embeddings** | (via Ollama) | — | nomic-embed-text served by Ollama |
| **Document parsing** | pymupdf4llm | >=0.0.10 | PDF → Markdown |
| | pymupdf | >=1.24.0 | PDF engine (pymupdf4llm dependency) |
| | trafilatura | >=1.8.0 | HTML content extraction |
| | beautifulsoup4 | >=4.12.0 | HTML parsing |
| | mammoth | >=1.8.0 | DOCX → Markdown |
| **Web server** | fastapi | >=0.110.0 | HTTP API + SSE |
| | uvicorn[standard] | >=0.29.0 | ASGI server |
| | jinja2 | >=3.1.0 | HTML templates |
| **MCP** | mcp | >=1.0.0 | MCP server SDK |
| **Database** | aiosqlite | >=0.20.0 | Async SQLite access |
| **HTTP client** | httpx | >=0.27.0 | Async HTTP calls to Ollama |
| **Logging** | structlog | >=24.1.0 | Structured JSON logging |
| **Config** | pydantic-settings | >=2.2.0 | Settings management |
| | pyyaml | >=6.0 | YAML config file parsing |
| **Dev/test** | pytest | >=8.0 | Testing |
| | pytest-asyncio | >=0.23.0 | Async test support |
| | ruff | >=0.3.0 | Linting + formatting |

### 8.2 Framework Decision: LangGraph

**Decision: LangGraph over CrewAI, AutoGen, or custom.**

| Criterion | LangGraph | CrewAI | AutoGen | Custom |
|-----------|-----------|--------|---------|--------|
| Execution model clarity | Explicit graph topology | Implicit (roles + goals) | Actor model | Manual |
| Ollama support | Excellent (langchain-ollama) | Good (via LiteLLM) | Good (autogen-ext) | Direct HTTP |
| Retry/cycle support | First-class (conditional edges) | Limited | Manual | Manual |
| State checkpointing | Built-in (SqliteSaver) | No | Manual | Manual |
| Streaming events | Excellent (.astream_events) | Basic verbose | Basic | Manual |
| Learning value | High (teaches graph/state machines) | Medium | Medium | Highest |
| Community / docs | Large, active | Growing | Large but fragmented | N/A |
| Portfolio signal | Strong (industry-adopted) | Good | Good | Shows fundamentals |

**Key reasons for LangGraph:**
1. **Explicit control**: The graph model maps directly to our pipeline (Planner→Gatherer→Analyst→Critic→Writer with conditional edges). No magic.
2. **Built-in checkpointing**: `SqliteSaver` persists pipeline state after each node. If the process crashes mid-pipeline, we can resume from the last checkpoint.
3. **Streaming**: `.astream_events()` gives us token-level and node-level events — exactly what we need for the SSE-based progress UI.
4. **Conditional edges**: The Critic→Analyst retry loop is a one-liner with `add_conditional_edges`.

### 8.3 Async Processing

**Decision: `asyncio.create_task()` inside FastAPI. No Celery, no Redis.**

**Rationale:** This is a single-machine, single-process application. Celery requires a broker (Redis/RabbitMQ) — that's an extra service to install and manage for zero benefit at this scale.

**Pattern:**
```python
# Pipeline jobs managed as asyncio tasks
active_pipelines: dict[str, asyncio.Task] = {}

@app.post("/api/research")
async def start_research(request: ResearchRequest):
    job_id = str(uuid.uuid4())
    task = asyncio.create_task(
        run_pipeline(job_id, request.topic, request.depth)
    )
    active_pipelines[job_id] = task
    return {"job_id": job_id}
```

**Limitation:** Tasks don't survive server restarts. This is acceptable for a local tool. The LangGraph checkpointer allows manual resumption of interrupted pipelines if needed.

---

## 9. Configuration Schema

All configuration lives in a single `config.yaml` at the project root, loaded via Pydantic Settings.

```yaml
# config.yaml — ResearchForge Configuration

ollama:
  base_url: "http://ollama:11434"   # Docker service name (use "http://localhost:11434" outside Docker)
  request_timeout_seconds: 120      # Per-model inference timeout
  keep_alive: "5m"                  # How long to keep models loaded

models:
  planner: "deepseek-r1:14b"
  gatherer: "qwen2.5:7b"
  analyst: "qwen2.5:14b"
  critic: "deepseek-r1:7b"
  writer: "mistral-nemo:12b"
  embedding: "nomic-embed-text"
  eval_judge: "qwen2.5:14b"        # Model used for LLM-as-judge eval

  # Fallback models (used when primary model fails)
  fallbacks:
    planner: "qwen2.5:7b"
    analyst: "qwen2.5:7b"
    critic: "qwen2.5:7b"
    writer: "qwen2.5:7b"

chunking:
  chunk_size: 1500                  # Characters (approx 375 tokens)
  chunk_overlap: 200                # Characters
  separators:
    - "\n\n"
    - "\n"
    - ". "
    - " "

retrieval:
  vector_candidates: 20             # Candidates from vector search
  bm25_candidates: 20              # Candidates from BM25 search
  final_top_k: 5                   # Results after RRF fusion
  embedding_prefix_document: "search_document: "
  embedding_prefix_query: "search_query: "

pipeline:
  max_critic_retries: 2
  quality_threshold_for_corpus: 0.6  # Min eval score to ingest briefing into RAG
  max_concurrent_pipelines: 1

storage:
  data_dir: "./data"                 # Base directory for all persistent data
  vector_db_path: "./data/lancedb"
  metadata_db_path: "./data/metadata.db"
  checkpoints_db_path: "./data/checkpoints.db"
  briefings_dir: "./data/briefings"
  logs_dir: "./logs"

web:
  host: "0.0.0.0"                   # Bind all interfaces inside container (exposed via Docker port mapping)
  port: 8000

mcp:
  transport: "stdio"                 # "stdio" or "sse"
  sse_host: "0.0.0.0"              # Inside container; exposed to host only via Docker
  sse_port: 8001
```

**Pydantic Settings model:**

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class OllamaConfig(BaseSettings):
    base_url: str = "http://ollama:11434"  # Docker service name
    request_timeout_seconds: int = 120
    keep_alive: str = "5m"

class ModelsConfig(BaseSettings):
    planner: str = "deepseek-r1:14b"
    gatherer: str = "qwen2.5:7b"
    analyst: str = "qwen2.5:14b"
    critic: str = "deepseek-r1:7b"
    writer: str = "mistral-nemo:12b"
    embedding: str = "nomic-embed-text"
    eval_judge: str = "qwen2.5:14b"

class Settings(BaseSettings):
    ollama: OllamaConfig = OllamaConfig()
    models: ModelsConfig = ModelsConfig()
    # ... etc
```

---

## 10. Error Handling Strategy

### 10.1 Error Categories and Responses

| Error Category | Example | Response |
|----------------|---------|----------|
| **Ollama container down** | Connection refused on `ollama:11434` | Fail fast. Display "Ollama container not running — run `docker compose up ollama`" in UI. All tools return error. |
| **Ollama no GPU detected** | Container starts but GPU not available | Log warning at startup. Ollama falls back to CPU automatically. Display "Running on CPU (slower)" in UI. |
| **Model not pulled** | `qwen2.5:14b` not available | Check at startup, list missing models. Auto-pull via `docker compose exec ollama ollama pull <model>`. |
| **Inference timeout** | Model takes >120s | Retry once. If still timing out, fall back to smaller model. |
| **Malformed output** | Agent returns invalid JSON | Retry with schema reminder (up to 2 times). Use `format: {schema}` on retry. |
| **Empty/irrelevant output** | Agent returns off-topic text | Log, increment error count, continue pipeline with degraded quality flag. |
| **RAG index corruption** | LanceDB read error | Fall back to creating a new index. Old data in SQLite can be re-indexed. |
| **Disk full** | Write failure | Fail the current operation. Display storage warning in UI. |
| **Concurrent access** | Two pipelines write to same table | LanceDB handles concurrent reads. Writes use an asyncio Lock. |

### 10.2 Graceful Degradation Ladder

```
Normal operation (all agents succeed, GPU-accelerated)
        │
        ▼ (GPU unavailable in Ollama container)
CPU fallback: Ollama runs on CPU, 5-10x slower but functional
        │
        ▼ (agent fails after retries)
Degraded: Skip failed agent, flag in output
        │
        ▼ (primary model unavailable)
Fallback: Use 7B model for all roles
        │
        ▼ (Ollama container down)
Offline: RAG queries still work (vector search is in app container)
         Pipeline unavailable until Ollama container restarts
```

### 10.3 Health Check Endpoint

```python
@app.get("/api/health")
async def health_check():
    return {
        "ollama": await check_ollama(),      # True/False + latency to ollama:11434
        "ollama_gpu": await check_ollama_gpu(),  # True/False — is GPU detected?
        "models": await list_loaded_models(), # Which models are pulled
        "vector_db": check_lancedb(),         # True/False + doc count
        "metadata_db": check_sqlite(),        # True/False
        "disk_space_gb": get_free_disk(),     # Available disk in data volume
    }
```

---

## 11. Agent Prompt Design Guidelines

### 11.1 Principles for Small Models (7B-14B)

Small models need **explicit, structured, constrained** prompts. They don't handle ambiguity well.

**Rule 1: One task per prompt.** Don't ask a 7B model to "analyze, critique, and summarize." Each agent does exactly one thing.

**Rule 2: Specify output format exactly.** Always provide a JSON schema or Markdown template. Use Ollama's `format` parameter to enforce JSON.

**Rule 3: Provide examples.** One-shot or few-shot prompting dramatically improves output quality with small models. Include one example of the expected input→output transformation.

**Rule 4: Keep system prompts under 500 tokens.** Small models' instruction-following degrades with long system prompts. Be concise.

**Rule 5: Use delimiters.** Clearly separate the context, instructions, and input with XML-style tags or markdown headers:
```
<context>
{retrieved chunks}
</context>

<instructions>
Analyze the above context and produce findings.
</instructions>

<input>
Research question: {question}
</input>
```

**Rule 6: Constrain the output space.** Instead of "rate the quality," say "rate the quality as one of: HIGH, MEDIUM, LOW." Instead of "list issues," say "list exactly 3-5 issues."

### 11.2 Prompt Templates

**Planner Agent System Prompt:**
```
You are a research planner. Given a research question, decompose it into 3-7 specific sub-questions that, when answered, would provide a comprehensive understanding of the topic.

For each sub-question, specify:
- The sub-question itself
- What type of information is needed (facts, statistics, opinions, definitions)
- Priority (high, medium, low)

Respond with valid JSON matching this schema:
{
  "sub_questions": [
    {
      "question": "string",
      "info_type": "string",
      "priority": "high|medium|low"
    }
  ]
}
```

**Critic Agent System Prompt:**
```
You are an adversarial research critic. Review the following analysis and check for:
1. Claims not supported by the provided evidence
2. Logical gaps or non-sequiturs
3. Missing important perspectives
4. Factual inconsistencies between sections
5. Bias toward one viewpoint without acknowledging others

Respond with valid JSON:
{
  "verdict": "pass" or "revise",
  "issues": ["issue 1", "issue 2"],
  "unsupported_claims": ["claim 1"],
  "missing_perspectives": ["perspective 1"],
  "overall_assessment": "one paragraph summary"
}

If you find no significant issues, set verdict to "pass" with an empty issues list.
If you find issues, set verdict to "revise" and list them specifically.
```

### 11.3 Context Window Budget

For each agent, the prompt budget is allocated as:

```
System prompt:        ~300-500 tokens
Retrieved context:    ~1500-2000 tokens (5 chunks × 300-400 tokens each)
User/task input:      ~200-500 tokens
Reserved for output:  ~1000-2000 tokens
──────────────────────────────────────
Total:                ~3000-5000 tokens per agent call
```

This fits comfortably within even a 4K context window, though most recommended models support 8K-32K. Keeping prompts small improves quality — small models perform better with less context.

---

## 12. Project Directory Structure

```
researchforge/
├── docker-compose.yml             # Main Compose file (GPU-enabled Ollama)
├── docker-compose.cpu.yml         # Override for CPU-only hosts (no GPU)
├── Dockerfile                     # App container (Python + FastAPI + agents)
├── .dockerignore                  # Exclude data/, logs/, .git, etc.
├── config.yaml                    # User configuration
├── pyproject.toml                 # Python project metadata + dependencies
├── README.md                      # Setup and usage instructions
│
├── scripts/
│   ├── setup_models.sh            # Pull all required Ollama models (runs inside ollama container)
│   ├── seed_corpus.py             # Seed the RAG corpus with sample docs
│   └── mcp_wrapper.sh            # Claude Desktop MCP wrapper (docker compose exec)
│
├── src/
│   └── researchforge/
│       ├── __init__.py
│       ├── __main__.py            # CLI entry point
│       ├── config.py              # Pydantic Settings models
│       │
│       ├── agents/                # Multi-agent pipeline
│       │   ├── __init__.py
│       │   ├── graph.py           # LangGraph pipeline definition
│       │   ├── state.py           # PipelineState TypedDict
│       │   ├── planner.py         # Planner agent node
│       │   ├── gatherer.py        # Gatherer agent node
│       │   ├── analyst.py         # Analyst agent node
│       │   ├── critic.py          # Critic agent node
│       │   ├── writer.py          # Writer agent node
│       │   └── prompts/           # Prompt templates (Jinja2 or plain text)
│       │       ├── planner.txt
│       │       ├── gatherer.txt
│       │       ├── analyst.txt
│       │       ├── critic.txt
│       │       └── writer.txt
│       │
│       ├── rag/                   # RAG subsystem
│       │   ├── __init__.py
│       │   ├── ingest.py          # Document parsing + chunking + embedding
│       │   ├── parsers.py         # PDF, HTML, DOCX, MD parsers
│       │   ├── chunker.py         # Chunking strategies
│       │   ├── embeddings.py      # Ollama embedding client
│       │   ├── store.py           # LanceDB vector store wrapper
│       │   └── retriever.py       # Hybrid search (vector + BM25 + RRF)
│       │
│       ├── mcp_server/            # MCP server
│       │   ├── __init__.py
│       │   └── server.py          # FastMCP tool definitions
│       │
│       ├── eval/                  # Eval engine
│       │   ├── __init__.py
│       │   ├── runner.py          # Eval suite runner
│       │   ├── retrieval_eval.py  # Retrieval quality metrics
│       │   ├── agent_eval.py      # Agent output rubric scoring
│       │   ├── e2e_eval.py        # End-to-end briefing evaluation
│       │   ├── benchmark.py       # Model benchmarking
│       │   └── judge.py           # LLM-as-judge implementation
│       │
│       ├── web/                   # Web UI
│       │   ├── __init__.py
│       │   ├── app.py             # FastAPI application
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── research.py    # Pipeline endpoints
│       │   │   ├── corpus.py      # RAG corpus endpoints
│       │   │   ├── briefings.py   # Briefing endpoints
│       │   │   ├── eval.py        # Eval dashboard endpoints
│       │   │   └── config.py      # Configuration endpoints
│       │   ├── templates/         # Jinja2 HTML templates
│       │   │   ├── base.html
│       │   │   ├── dashboard.html
│       │   │   ├── briefing.html
│       │   │   ├── corpus.html
│       │   │   ├── eval.html
│       │   │   └── config.html
│       │   └── static/            # CSS, HTMX, minimal JS
│       │       ├── style.css
│       │       └── htmx.min.js
│       │
│       └── db/                    # Database layer
│           ├── __init__.py
│           ├── models.py          # SQLite table schemas
│           └── repository.py      # Data access methods
│
├── eval/                          # Eval datasets and results (not in src)
│   ├── datasets/
│   │   ├── retrieval_test.jsonl
│   │   ├── critic_test.jsonl
│   │   └── reference_briefings/
│   └── results/
│       ├── retrieval_scores.jsonl
│       ├── agent_scores.jsonl
│       ├── e2e_scores.jsonl
│       └── model_benchmark.json
│
├── data/                          # Persistent data (gitignored)
│   ├── lancedb/                   # Vector store
│   ├── metadata.db                # SQLite metadata
│   ├── checkpoints.db             # LangGraph state checkpoints
│   └── briefings/                 # Completed briefing files
│
├── logs/                          # Application logs (gitignored)
│   └── pipeline.jsonl
│
├── tests/
│   ├── conftest.py
│   ├── test_agents/
│   │   ├── test_planner.py
│   │   ├── test_gatherer.py
│   │   ├── test_analyst.py
│   │   ├── test_critic.py
│   │   └── test_writer.py
│   ├── test_rag/
│   │   ├── test_ingest.py
│   │   ├── test_parsers.py
│   │   ├── test_chunker.py
│   │   └── test_retriever.py
│   ├── test_mcp/
│   │   └── test_server.py
│   ├── test_eval/
│   │   └── test_runner.py
│   └── test_web/
│       └── test_routes.py
│
└── .gitignore
```

---

## 13. Docker & Containerization

### 13.1 Container Architecture

ResearchForge runs as **two containers** orchestrated by Docker Compose:

| Container | Image | Purpose | Volumes | Ports |
|-----------|-------|---------|---------|-------|
| **app** | Built from `./Dockerfile` | Python app (agents, RAG, web UI, MCP) | `./data:/app/data`, `./logs:/app/logs`, `./config.yaml:/app/config.yaml` | `8000:8000` (web UI) |
| **ollama** | `ollama/ollama:latest` | LLM inference server with GPU | `ollama_models:/root/.ollama` | `11434:11434` (optional, for debugging) |

**Why two containers instead of one:**
- Ollama has its own release cycle and GPU driver dependencies — keeping it separate avoids bloating the app image
- The Ollama container can be restarted independently without losing pipeline state
- Model files (~5-10 GB each) persist in a named Docker volume and survive container rebuilds
- The official `ollama/ollama` image handles GPU detection and CUDA automatically

### 13.2 docker-compose.yml (GPU-Enabled)

```yaml
# docker-compose.yml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: researchforge-ollama
    ports:
      - "11434:11434"          # Optional: expose to host for debugging
    volumes:
      - ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all       # Pass through all GPUs
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s

  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: researchforge-app
    ports:
      - "8000:8000"            # Web UI
    volumes:
      - ./data:/app/data       # Persistent data (LanceDB, SQLite, briefings)
      - ./logs:/app/logs       # Application logs
      - ./config.yaml:/app/config.yaml  # User configuration
      - ./eval:/app/eval       # Eval datasets and results
    environment:
      - RESEARCHFORGE_OLLAMA__BASE_URL=http://ollama:11434
    depends_on:
      ollama:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:8000/api/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  ollama_models:               # Named volume — persists Ollama models across rebuilds
```

### 13.3 docker-compose.cpu.yml (CPU-Only Override)

For hosts without an NVIDIA GPU, use this override file to remove the GPU reservation:

```yaml
# docker-compose.cpu.yml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices: []          # Override: no GPU
```

**Usage:**
```bash
# GPU host (default)
docker compose up -d

# CPU-only host
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

### 13.4 Dockerfile (App Container)

```dockerfile
# Dockerfile
FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies for document parsing (pymupdf, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[all]"

# Copy application source
COPY src/ src/
COPY config.yaml ./
COPY eval/ eval/
COPY scripts/ scripts/

# Create data and log directories
RUN mkdir -p data logs

# Expose web UI port
EXPOSE 8000

# Default: start the web server
CMD ["python", "-m", "researchforge", "serve"]
```

### 13.5 .dockerignore

```
.git
.github
__pycache__
*.pyc
.pytest_cache
.ruff_cache
data/
logs/
*.egg-info
.env
.venv
venv
node_modules
```

### 13.6 GPU Passthrough — NVIDIA Container Toolkit

The Ollama container requires the **NVIDIA Container Toolkit** to access the host GPU. This is a one-time setup on the host machine.

**How it works:**
1. The host has NVIDIA GPU drivers installed (driver version >= 535)
2. The NVIDIA Container Toolkit (`nvidia-ctk`) hooks into the Docker runtime
3. `docker-compose.yml` specifies `deploy.resources.reservations.devices` with `driver: nvidia`
4. Docker automatically mounts the GPU device and CUDA libraries into the Ollama container
5. Ollama detects the GPU and offloads model layers to VRAM

**Installation (one-time, on host):**
```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verify GPU is accessible in container:**
```bash
docker compose up -d ollama
docker compose exec ollama nvidia-smi
# Should show your GPU model, driver version, and CUDA version

docker compose exec ollama ollama run qwen2.5:7b "Say hello"
# Check logs: should show "using GPU" or similar
```

**Troubleshooting GPU detection:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nvidia-smi` not found in container | NVIDIA Container Toolkit not installed | Install toolkit (see above) |
| `could not select device driver "nvidia"` | Docker runtime not configured | Run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| Ollama runs on CPU despite GPU being present | Model too large for VRAM | Use smaller quantization or smaller model. Check `docker compose logs ollama` for memory errors. |
| `nvidia-smi` works but Ollama still uses CPU | Ollama image issue | Ensure using `ollama/ollama:latest` (not an old tag) |

### 13.7 Volume Strategy

| Volume | Type | Mount | Purpose | Survives `docker compose down`? |
|--------|------|-------|---------|------|
| `ollama_models` | Named volume | `/root/.ollama` in ollama container | Ollama model files (5-10 GB each) | Yes (named volumes persist) |
| `./data` | Bind mount | `/app/data` in app container | LanceDB, SQLite, briefings | Yes (host directory) |
| `./logs` | Bind mount | `/app/logs` in app container | Application logs | Yes (host directory) |
| `./config.yaml` | Bind mount | `/app/config.yaml` in app container | User configuration | Yes (host file) |
| `./eval` | Bind mount | `/app/eval` in app container | Eval datasets and results | Yes (host directory) |

**Why bind mounts for data/logs/eval:** These are user-visible files that should be easy to inspect, back up, and version control (eval datasets). Bind mounts make them accessible on the host filesystem directly.

**Why a named volume for Ollama models:** Model files are large binary blobs managed entirely by Ollama. They don't need to be user-visible on the host. A named volume is cleaner and avoids permission issues.

### 13.8 Model Management

Models are pulled inside the running Ollama container. The `setup_models.sh` script handles this:

```bash
#!/bin/bash
# scripts/setup_models.sh — Pull all required Ollama models
# Run with: docker compose exec ollama bash /app/scripts/setup_models.sh
# Or:       docker compose run --rm ollama bash -c "ollama pull ..."

set -e

echo "Pulling ResearchForge models..."

MODELS=(
    "deepseek-r1:14b"
    "qwen2.5:14b"
    "qwen2.5:7b"
    "deepseek-r1:7b"
    "mistral-nemo:12b"
    "nomic-embed-text"
)

for model in "${MODELS[@]}"; do
    echo "Pulling $model..."
    ollama pull "$model"
done

echo "All models pulled. Verify with: ollama list"
ollama list
```

**Usage after `docker compose up`:**
```bash
docker compose exec ollama bash /scripts/setup_models.sh
```

Alternatively, model pulling is triggered from the app's startup health check — missing models are auto-pulled when first needed (with a warning in the UI about the download delay).

### 13.9 MCP Server in Docker

For Claude Desktop integration, the MCP server runs inside the app container. A wrapper script on the host handles the `docker compose exec`:

```bash
#!/bin/bash
# scripts/mcp_wrapper.sh — Wrapper for Claude Desktop MCP integration
# Place this path in Claude Desktop's MCP config

exec docker compose -f /path/to/researchforge/docker-compose.yml \
    exec -T app python -m researchforge mcp
```

**Claude Desktop configuration:**
```json
{
  "mcpServers": {
    "researchforge": {
      "command": "/path/to/researchforge/scripts/mcp_wrapper.sh"
    }
  }
}
```

The `-T` flag disables pseudo-TTY allocation, which is required for stdio-based MCP transport.

### 13.10 Development Workflow

For local development (editing code with hot reload):

```yaml
# docker-compose.override.yml (git-ignored, for local dev only)
services:
  app:
    build:
      context: .
    volumes:
      - ./src:/app/src          # Mount source code for live editing
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config.yaml:/app/config.yaml
      - ./eval:/app/eval
    command: ["python", "-m", "researchforge", "serve", "--reload"]
```

```bash
# Dev mode: code changes auto-reload
docker compose up -d

# Run tests inside container
docker compose exec app pytest -m "not ollama"
docker compose exec app pytest  # Full suite (Ollama must be healthy)

# Access app shell for debugging
docker compose exec app bash

# View Ollama logs
docker compose logs -f ollama

# View app logs
docker compose logs -f app
```

---

## Appendix A: Minimum Viable Setup (16GB RAM)

Pull only the minimum models:
```bash
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull nomic-embed-text
```

Set in `config.yaml`:
```yaml
models:
  planner: "qwen2.5:7b"
  gatherer: "qwen2.5:7b"
  analyst: "qwen2.5:7b"
  critic: "qwen2.5:7b"
  writer: "qwen2.5:7b"
  embedding: "nomic-embed-text"
```

## Appendix B: Quick Start

```bash
# 1. Clone
git clone <repo-url> && cd researchforge

# 2. Start the stack (GPU)
docker compose up -d

# 3. Pull models (first time only — takes a while)
docker compose exec ollama bash scripts/setup_models.sh

# 4. Seed sample corpus
docker compose exec app python scripts/seed_corpus.py

# 5. Open the UI
open http://localhost:8000

# CPU-only host:
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```
