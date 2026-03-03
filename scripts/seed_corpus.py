#!/usr/bin/env python3
"""Seed the ResearchForge corpus with sample documents for demo purposes.

Usage:
    python scripts/seed_corpus.py
    # or inside Docker:
    docker compose exec app python scripts/seed_corpus.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from textwrap import dedent

from researchforge.config import get_settings
from researchforge.db.repository import Repository
from researchforge.rag.ingest import ingest_file
from researchforge.rag.store import VectorStore

# ---------------------------------------------------------------------------
# Sample documents — concise Markdown articles on topics the eval suite and
# reference briefings cover.  These give the RAG corpus enough material for
# meaningful retrieval during demos, research queries, and eval runs.
# ---------------------------------------------------------------------------

SAMPLE_DOCS: dict[str, str] = {
    "rag_overview.md": dedent("""\
        # Retrieval Augmented Generation: An Overview

        ## Introduction

        Retrieval Augmented Generation (RAG) enhances large language model
        outputs by grounding them in retrieved factual information. Instead of
        relying solely on parametric knowledge, RAG dynamically retrieves
        relevant documents from an external corpus and injects them into the
        prompt as context [1].

        ## Architecture

        A typical RAG pipeline has three stages:

        1. **Indexing** — Parse documents, split into chunks, embed with a
           vector model, and store in a vector database.
        2. **Retrieval** — Given a query, find the most relevant chunks via
           vector similarity, keyword (BM25) search, or a hybrid of both.
        3. **Generation** — Provide retrieved chunks as context to an LLM,
           which generates a grounded response.

        ## Benefits

        - Reduced hallucination by providing factual evidence [1].
        - Up-to-date knowledge without retraining.
        - Source attribution — responses cite specific documents.
        - Cost-effective alternative to fine-tuning.

        ## Challenges

        - Retrieval quality directly limits generation quality.
        - Chunking strategy affects what the model can "see."
        - Context window limits constrain how many chunks fit.
        - Embedding model choice impacts semantic matching accuracy.

        ## References

        [1] Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive
            NLP Tasks," NeurIPS 2020.
        [2] Gao et al., "Retrieval-Augmented Generation for Large Language Models:
            A Survey," 2024.
    """),

    "hybrid_search.md": dedent("""\
        # Hybrid Search in RAG Systems

        ## Overview

        Hybrid search combines vector similarity search with keyword-based
        search (typically BM25) to improve retrieval quality. Neither approach
        alone is optimal: vector search captures semantic meaning but may miss
        exact keyword matches, while BM25 excels at exact term matching but
        misses semantically similar content [1].

        ## How Hybrid Search Works

        1. **Vector search** — The query is embedded and compared against chunk
           embeddings using cosine similarity or L2 distance.
        2. **BM25 search** — A full-text index (e.g., Tantivy) scores documents
           by term frequency and inverse document frequency.
        3. **Reciprocal Rank Fusion (RRF)** — Results from both methods are
           merged using RRF: `score = sum(1 / (k + rank))` across the two
           ranked lists. This avoids score normalization issues.

        ## Performance

        Studies show hybrid search outperforms vector-only retrieval by 10-20%
        on precision for typical RAG workloads [2]. The gains are largest when
        queries contain domain-specific terms or proper nouns.

        ## Implementation Notes

        - LanceDB supports hybrid search natively via its FTS + vector index.
        - The `k` constant in RRF is typically set to 60.
        - Candidate sets of 20 from each method, merged to final top-5, is a
          common configuration.

        ## References

        [1] Chen et al., "Benchmarking Retrieval Strategies for RAG," 2024.
        [2] Ma et al., "Hybrid Dense-Sparse Retrieval," SIGIR 2023.
    """),

    "chunking_strategies.md": dedent("""\
        # Document Chunking Strategies for RAG

        ## Why Chunking Matters

        Chunking divides source documents into smaller segments for embedding
        and retrieval. The chunk size and overlap directly affect retrieval
        precision and the quality of generated responses [1].

        ## Common Strategies

        ### Fixed-Size Chunking
        Split text every N characters with M characters of overlap. Simple and
        fast, but ignores document structure.

        ### Recursive Character Splitting
        Split on a hierarchy of separators (paragraphs → sentences → words).
        This preserves semantic boundaries better than fixed-size splitting.

        ### Structure-Aware Chunking
        Use document structure (Markdown headers, HTML tags, PDF sections) to
        create chunks that respect logical boundaries. A Markdown pre-pass
        splits on headers, then a recursive splitter handles oversized sections.

        ### Semantic Chunking
        Embed consecutive sentences and split where cosine similarity drops
        below a threshold. Produces semantically coherent chunks but is slower.

        ## Optimal Parameters

        - **Chunk size**: 500-1500 characters (~125-375 tokens) is optimal for
          most use cases [2].
        - **Overlap**: 10-15% of chunk size prevents information loss at
          boundaries.
        - **Metadata**: Preserving section headers as chunk metadata improves
          retrieval context.

        ## References

        [1] Pinecone, "Chunking Strategies for LLM Applications," 2024.
        [2] Langchain Documentation, "Text Splitters," 2024.
    """),

    "vector_databases.md": dedent("""\
        # Vector Databases for RAG Systems

        ## Purpose

        Vector databases store high-dimensional embeddings and support fast
        approximate nearest neighbor (ANN) search. In RAG systems, they serve
        as the retrieval backbone [1].

        ## Popular Options

        ### LanceDB
        Embedded (no server), built on the Lance columnar format. Supports
        hybrid search (vector + full-text via Tantivy). Open source, with
        native Python integration. Ideal for local-first applications.

        ### ChromaDB
        Embedded vector database with a simple Python API. Good for
        prototyping but limited hybrid search support.

        ### Pinecone
        Cloud-hosted, fully managed. High performance but requires internet
        connectivity and a paid plan.

        ### Weaviate
        Open source with cloud options. Supports hybrid search, multi-modal
        data, and GraphQL queries.

        ### Qdrant
        Open source, Rust-based. Fast ANN search with payload filtering.

        ## Key Considerations

        - **Embedding dimensionality**: 768-dim (e.g., nomic-embed-text) is a
          good balance of quality and speed.
        - **Index type**: HNSW is the most common ANN index for low-latency
          retrieval.
        - **Filtering**: Metadata filtering (e.g., by source type or date) is
          essential for large corpora.
        - **Scalability**: Embedded DBs (LanceDB, Chroma) work well up to ~10M
          vectors; larger corpora need distributed solutions.

        ## References

        [1] Bruch et al., "An Introduction to Vector Databases," 2023.
        [2] LanceDB Documentation, https://lancedb.com/docs.
    """),

    "embedding_models.md": dedent("""\
        # Embedding Models for RAG

        ## Role of Embeddings

        Embedding models convert text into dense vector representations that
        capture semantic meaning. In RAG, both documents and queries are
        embedded so that semantically similar content can be retrieved via
        vector similarity search [1].

        ## Common Models

        | Model | Dimensions | Context | Notes |
        |-------|-----------|---------|-------|
        | nomic-embed-text | 768 | 8192 tokens | Open, runs locally via Ollama |
        | all-MiniLM-L6 | 384 | 256 tokens | Small, fast, lower quality |
        | BGE-large-en | 1024 | 512 tokens | High quality, larger |
        | text-embedding-3-small | 1536 | 8191 tokens | OpenAI, cloud-only |
        | mxbai-embed-large | 1024 | 512 tokens | Open, high MTEB scores |

        ## Prefixed Embeddings

        Some models (including nomic-embed-text) use task-specific prefixes:
        - `search_document:` prefix for document chunks during indexing.
        - `search_query:` prefix for user queries during retrieval.
        This asymmetric approach improves retrieval quality [2].

        ## Choosing a Model

        - **384-dim models** are fast but underperform on nuanced queries.
        - **768-dim models** (nomic-embed-text) offer the best speed/quality
          tradeoff for local deployments.
        - **1024+ dim models** provide marginal gains at higher compute cost.

        ## References

        [1] Muennighoff et al., "MTEB: Massive Text Embedding Benchmark," 2023.
        [2] Nomic AI, "nomic-embed-text Technical Report," 2024.
    """),

    "multi_agent_systems.md": dedent("""\
        # Multi-Agent Systems for Research

        ## Overview

        Multi-agent systems decompose complex tasks into subtasks handled by
        specialized agents. In research applications, different agents handle
        planning, information gathering, analysis, quality control, and
        writing [1].

        ## Agent Roles

        ### Planner
        Decomposes a broad research question into specific sub-questions and
        identifies information needs. Outputs a structured research plan.

        ### Gatherer
        Retrieves relevant evidence from the corpus for each sub-question.
        May perform multiple retrieval rounds with query reformulation.

        ### Analyst
        Synthesizes gathered evidence into structured findings. Identifies
        patterns, contradictions, and knowledge gaps across sources.

        ### Critic
        Reviews the analysis for logical errors, unsupported claims, and
        completeness. Issues a "pass" or "revise" verdict.

        ### Writer
        Produces the final briefing document from the reviewed analysis.
        Includes executive summary, key findings, and source citations.

        ## Orchestration

        LangGraph provides a state machine approach where:
        - State flows through a directed graph of agent nodes.
        - Conditional edges enable feedback loops (e.g., Critic → Analyst).
        - Checkpointing allows crash recovery mid-pipeline.

        ## Benefits

        - Each agent uses the model best suited to its task.
        - The critic feedback loop improves output quality.
        - Pipeline traces provide full transparency into the research process.

        ## References

        [1] Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via
            Multi-Agent Conversation," 2023.
        [2] LangGraph Documentation, https://langchain-ai.github.io/langgraph/.
    """),

    "llm_evaluation.md": dedent("""\
        # Evaluating LLM-Generated Research Output

        ## Why Evaluation Matters

        Without systematic evaluation, it is impossible to know whether changes
        to prompts, models, or retrieval settings improve or degrade output
        quality. An eval engine provides objective measurement [1].

        ## Evaluation Dimensions

        ### Retrieval Quality
        - **Precision@K**: Fraction of retrieved chunks that are relevant.
        - **Recall@K**: Fraction of all relevant chunks that were retrieved.
        - **MRR (Mean Reciprocal Rank)**: How high the first relevant result
          appears in the ranked list.

        ### Agent Output Quality (LLM-as-Judge)
        An LLM scores outputs on a rubric:
        - **Structural validity** (0.15 weight): Proper Markdown, sections.
        - **Relevance** (0.30 weight): Content addresses the research question.
        - **Completeness** (0.25 weight): Key aspects are covered.
        - **Coherence** (0.20 weight): Logical flow and consistency.
        - **Conciseness** (0.10 weight): No unnecessary repetition.

        Three independent scoring runs with median aggregation reduce variance.

        ### End-to-End Quality
        Run the full pipeline on reference topics and compare outputs to
        reference briefings. Measures overall system performance.

        ### Regression Detection
        Compare current scores to a rolling average of the last 5 runs. Flag
        any metric that drops more than 10% as a regression.

        ## Heuristic Scoring

        Lightweight checks that run without an LLM:
        - Word count > 200
        - Citation count (numbered references or [Source:] patterns)
        - Section count (Markdown headers)
        - Readability (average sentence length)

        ## References

        [1] Zheng et al., "Judging LLM-as-a-Judge," NeurIPS 2023.
        [2] RAGAS: Evaluation framework for RAG, https://docs.ragas.io/.
    """),

    "prompt_engineering.md": dedent("""\
        # Prompt Engineering for Research Agents

        ## Principles

        Effective prompts for research agents follow several key principles:

        ### Role Definition
        Assign a clear persona: "You are a research analyst specializing in..."
        This focuses the model's behavior and improves output consistency [1].

        ### Structured Output
        Request specific output formats (JSON schemas, Markdown templates).
        Use Ollama's `format` parameter with a JSON schema to enforce structure.

        ### Context Injection
        Provide retrieved evidence between clear delimiters:
        ```
        <evidence>
        [1] Source title — relevant excerpt...
        [2] Source title — relevant excerpt...
        </evidence>
        ```

        ### Chain of Thought
        For complex analysis, instruct the model to reason step-by-step:
        "First identify the key claims, then evaluate evidence for each claim,
        then note any contradictions."

        ## Common Pitfalls

        - **Over-prompting**: Too many instructions cause the model to miss
          important ones. Keep prompts focused.
        - **Ambiguous output format**: Without explicit format instructions,
          outputs vary unpredictably between runs.
        - **Missing context boundaries**: If evidence isn't clearly delimited,
          the model may confuse instructions with content.

        ## Temperature Settings

        - **Planning tasks**: Temperature 0.3-0.5 for focused decomposition.
        - **Analysis tasks**: Temperature 0.2-0.4 for factual synthesis.
        - **Writing tasks**: Temperature 0.5-0.7 for more natural prose.
        - **Critic tasks**: Temperature 0.1-0.3 for strict evaluation.

        ## References

        [1] Wei et al., "Chain-of-Thought Prompting," NeurIPS 2022.
        [2] Anthropic, "Prompt Engineering Guide," 2024.
    """),

    "mcp_protocol.md": dedent("""\
        # Model Context Protocol (MCP)

        ## Overview

        The Model Context Protocol (MCP) is an open standard for connecting AI
        assistants to external tools and data sources. It defines a JSON-RPC
        based interface for tool discovery, invocation, and result handling [1].

        ## Core Concepts

        ### Tools
        Functions that an AI assistant can call. Each tool has a name,
        description, and input schema (JSON Schema). Tools are discovered via
        the `tools/list` method.

        ### Resources
        Read-only data sources (files, database rows, API responses). Resources
        are identified by URIs.

        ### Prompts
        Reusable prompt templates that the server can provide to clients.

        ## Transports

        - **stdio**: Communication over stdin/stdout. Used by Claude Desktop
          and other local integrations.
        - **SSE (Server-Sent Events)**: HTTP-based streaming transport for
          web clients.

        ## Python SDK (FastMCP)

        The `mcp` Python package provides `FastMCP` for building servers:

        ```python
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("my-server")

        @mcp.tool()
        def my_tool(query: str) -> str:
            return f"Result for {query}"

        mcp.run(transport="stdio")
        ```

        ## Use Cases

        - Expose local tools to Claude Desktop for interactive research.
        - Connect a RAG corpus to any MCP-compatible AI assistant.
        - Bridge between AI assistants and domain-specific data systems.

        ## References

        [1] Anthropic, "Model Context Protocol Specification," 2024.
        [2] MCP Python SDK, https://github.com/modelcontextprotocol/python-sdk.
    """),

    "hallucination_reduction.md": dedent("""\
        # Reducing LLM Hallucinations with RAG

        ## The Problem

        Large language models generate text that sounds plausible but may be
        factually incorrect. This "hallucination" problem is especially
        dangerous in research contexts where accuracy is critical [1].

        ## How RAG Helps

        RAG reduces hallucination by:

        1. **Grounding in evidence**: The model generates from retrieved facts
           rather than parametric memory alone.
        2. **Constraining the answer space**: Context narrows what the model
           considers relevant.
        3. **Enabling verification**: Citations allow readers to check claims
           against source documents.

        ## Measured Impact

        - RAG systems reduce hallucination by 30-50% compared to base LLMs on
          factual QA benchmarks [1].
        - Hybrid retrieval (vector + BM25) further reduces hallucination by
          improving retrieval precision [2].
        - Multi-round retrieval with query reformulation catches edge cases
          missed by single-pass retrieval.

        ## Remaining Challenges

        - **Retrieval failures**: If relevant documents aren't in the corpus,
          RAG cannot help.
        - **Context window limits**: Large documents may not fit, forcing
          truncation.
        - **Conflicting sources**: When retrieved documents disagree, the model
          may arbitrarily choose one.
        - **Indirect hallucination**: The model may correctly retrieve evidence
          but draw incorrect conclusions from it.

        ## Best Practices

        - Use hybrid search to maximize retrieval recall.
        - Include a critic agent to catch unsupported claims.
        - Require explicit citation in generated output.
        - Monitor hallucination rates through systematic evaluation.

        ## References

        [1] Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive
            NLP Tasks," NeurIPS 2020.
        [2] Shuster et al., "Retrieval Augmentation Reduces Hallucination in
            Conversation," 2021.
    """),
}


async def main() -> int:
    """Write sample docs to a temp directory, then ingest them all."""
    settings = get_settings()
    store = VectorStore()
    repo = Repository(settings.storage.metadata_db_path)
    await repo.initialize()

    # Write sample documents to data/sample_docs/
    docs_dir = Path(settings.storage.data_dir) / "sample_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in SAMPLE_DOCS.items():
        (docs_dir / filename).write_text(content)

    print(f"Wrote {len(SAMPLE_DOCS)} sample documents to {docs_dir}")

    # Ingest each document
    try:
        total_chunks = 0
        ingested = 0
        for filename in sorted(SAMPLE_DOCS):
            path = docs_dir / filename
            result = await ingest_file(path, store, repo)
            if result.skipped:
                print(f"  {filename}: skipped ({result.reason})")
            else:
                print(f"  {filename}: {result.chunk_count} chunks")
                total_chunks += result.chunk_count
                ingested += 1

        print(f"\nSeeded corpus: {ingested} documents, {total_chunks} chunks total.")
        print(f"Corpus now has {store.count()} chunks.")
    finally:
        await repo.close()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
