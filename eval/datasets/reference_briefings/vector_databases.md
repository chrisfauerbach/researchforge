# Vector Databases for AI Applications

## Overview

Vector databases are specialized storage systems designed to efficiently store, index, and query high-dimensional vector embeddings. They are a critical component in modern AI applications including retrieval augmented generation (RAG), recommendation systems, and semantic search.

## Core Concepts

### Embeddings
Text, images, and other data are converted into dense numerical vectors (typically 384-1536 dimensions) using embedding models. These vectors capture semantic meaning — similar content produces vectors that are close together in the embedding space.

### Similarity Search
Vector databases support approximate nearest neighbor (ANN) search algorithms that efficiently find the most similar vectors to a query vector. Common distance metrics include cosine similarity, Euclidean distance, and dot product.

## Popular Vector Databases

- **LanceDB**: Embedded, serverless vector database built on the Lance columnar format. Supports hybrid search with built-in full-text search [1].
- **ChromaDB**: Open-source embedding database with a simple API, designed for LLM applications.
- **Pinecone**: Managed cloud vector database with high scalability.
- **Weaviate**: Open-source vector database with GraphQL API and module system.
- **Milvus**: Distributed vector database built for scalable similarity search.

## Hybrid Search

Modern vector databases increasingly support hybrid search — combining dense vector similarity with sparse keyword matching (BM25). Results from both approaches are typically fused using Reciprocal Rank Fusion (RRF) to produce a final ranked list that benefits from both semantic understanding and exact keyword matching.

## Selection Criteria

When choosing a vector database, key factors include:
- **Deployment model**: Embedded vs. client-server vs. managed cloud
- **Scale requirements**: Dataset size and query throughput
- **Search capabilities**: Vector-only vs. hybrid (vector + full-text)
- **Integration**: Language SDKs, framework compatibility

## References

[1] LanceDB Documentation, "Hybrid Search," 2024.
[2] Douze et al., "The Faiss Library," 2024.
