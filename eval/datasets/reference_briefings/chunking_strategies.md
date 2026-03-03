# Document Chunking Strategies for RAG

## Overview

Document chunking is the process of breaking large documents into smaller segments suitable for embedding and retrieval. The chunking strategy directly impacts retrieval quality — poorly chunked documents lead to irrelevant or incomplete results regardless of how good the embedding model or search algorithm is.

## Common Strategies

### Fixed-Size Chunking
Split text into chunks of a fixed character or token count with configurable overlap. Simple to implement but ignores document structure.

- **Typical sizes**: 500-2000 characters (125-500 tokens)
- **Overlap**: 10-20% of chunk size to preserve context across boundaries

### Recursive Character Splitting
Iteratively split on a hierarchy of separators (paragraphs, sentences, words) until chunks are within the size limit. Better preserves natural boundaries than fixed-size [1].

### Structure-Aware Splitting
Use document structure (Markdown headers, HTML tags, PDF sections) to define chunk boundaries. A two-pass approach works well:

1. **Pass 1**: Split on structural boundaries (headers, sections)
2. **Pass 2**: Apply size-constrained splitting within each section

This preserves section metadata (header hierarchy) in chunk metadata, enabling section-aware retrieval.

### Semantic Chunking
Use embedding similarity to detect topic shifts within a document. Group consecutive sentences with high similarity into chunks. More computationally expensive but produces semantically coherent chunks.

## Key Parameters

| Parameter | Typical Range | Impact |
|-----------|---------------|--------|
| Chunk size | 500-2000 chars | Smaller = more precise, larger = more context |
| Overlap | 50-200 chars | Prevents information loss at boundaries |
| Separators | Paragraphs, sentences | Controls split granularity |

## Best Practices

- Match chunk size to the context window and retrieval top-k: smaller chunks with higher top-k provides more granular coverage [2].
- Always include overlap to prevent information loss at chunk boundaries.
- Preserve metadata (source document, section headers, page numbers) for attribution.
- Test chunking parameters empirically with your specific corpus and queries.

## References

[1] LangChain Documentation, "Text Splitters," 2024.
[2] Pinecone, "Chunking Strategies for LLM Applications," 2024.
