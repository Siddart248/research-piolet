# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances Large Language Models by dynamically retrieving relevant facts from an external knowledge repository and injecting them into the model's context window before generating a response.

## Why RAG is Essential

Foundation models are static snapshots frozen at their training cutoff date. RAG bridges this gap by:
- Grounding answers in verified, domain-specific private data without costly model retraining.
- Mitigating hallucinations by providing explicit reference documents.
- Providing traceable provenance through document citations and source attribution.
- Allowing real-time updates: updating the knowledge base instantly updates model knowledge.

## The Standard RAG Pipeline

1. **Ingestion & Indexing**: Documents are loaded, split into smaller manageable text chunks, transformed into dense vector embeddings, and stored in an index or vector database.
2. **Retrieval**: At query time, the user's prompt is converted into an embedding. A similarity search (such as cosine similarity) identifies the top-k most semantically relevant chunks.
3. **Synthesis**: A prompt is constructed containing the original question and the retrieved chunks as authoritative context. The LLM generates a response constrained to this context.

## Main Limitations and Failure Modes of RAG

Despite its popularity, RAG architectures suffer from well-documented limitations:

1. **Retrieval Failures**: If the retrieval stage misses the necessary information due to keyword mismatch or vector semantic drift, the downstream LLM cannot answer correctly.
2. **Chunk Boundary Disconnection**: Chunking documents can sever vital context across chunk splits, resulting in fragmented pieces that miss surrounding nuance.
3. **Noise and Context Pollution**: Injecting irrelevant or borderline chunks clutters the context window, confusing the LLM and causing hallucinations.
4. **Multi-Hop Reasoning Challenges**: Standard similarity search struggles with questions requiring synthesis across disparate documents that do not share semantic similarity with the initial query.
5. **Stale Indices & Embedding Lag**: If source documents change but indices or embeddings are not refreshed, the system serves outdated or conflicting facts.
6. **Hallucination Under Partial Context**: When retrieved chunks provide only partial clues, models frequently fabricate the missing details to create an apparently coherent answer.
