# Retrieval Augmented Generation: Fundamentals

## Overview

Retrieval Augmented Generation (RAG) is an approach that enhances large language model (LLM) outputs by grounding them in retrieved factual information. Rather than relying solely on parametric knowledge encoded during training, RAG systems dynamically retrieve relevant documents from an external corpus and provide them as context to the model.

## How RAG Works

The RAG pipeline consists of three core stages:

1. **Indexing**: Documents are parsed, chunked into manageable segments, embedded into vector representations, and stored in a vector database.
2. **Retrieval**: Given a user query, the system retrieves the most relevant chunks using similarity search (vector, keyword, or hybrid).
3. **Generation**: The retrieved chunks are injected into the LLM prompt as context, and the model generates a response grounded in this evidence.

## Key Benefits

- **Reduced hallucination**: By providing factual context, RAG helps prevent the model from generating unsupported claims [1].
- **Up-to-date information**: The corpus can be updated without retraining the model.
- **Source attribution**: Responses can cite specific retrieved documents, enabling verification.
- **Cost efficiency**: RAG avoids the expense and complexity of fine-tuning large models.

## Challenges

- Chunking strategy significantly affects retrieval quality — chunks that are too small lose context, while chunks that are too large dilute relevance.
- Retrieval errors propagate to generation: if irrelevant documents are retrieved, the model may produce incorrect outputs.
- Hybrid search (combining vector and keyword approaches) generally outperforms either method alone.

## References

[1] Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020.
[2] Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey," 2024.
