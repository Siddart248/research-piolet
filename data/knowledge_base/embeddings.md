# Text Embeddings and Vector Similarity

Embeddings are dense numerical vector representations of text in a continuous, high-dimensional mathematical space (typically 768, 1536, or 3072 dimensions). Unlike traditional sparse lexical matching (e.g., BM25 or TF-IDF), embeddings capture latent semantic relationships, conceptual hierarchies, and contextual meanings.

## How Embeddings Capture Semantics

Trained transformer encoders project natural language into vectors such that text segments with related meanings occupy proximate positions in vector space. For example, "king - man + woman" approximates "queen", and "car" is closer to "automobile" than to "refrigerator".

## Measuring Vector Similarity: Cosine Similarity

The standard metric for comparing text embeddings is **cosine similarity**, which computes the cosine of the angle between two multi-dimensional vectors:

$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$

Key characteristics:
- Measures directional alignment rather than vector magnitude.
- Yields a score typically bounded between -1.0 and +1.0 (or 0.0 to 1.0 for normalized text embeddings).
- When vectors are unit-normalized ($\|A\| = 1$), cosine similarity simplifies directly to the dot product ($A \cdot B$), greatly accelerating computations with NumPy.

## Limitations of Dense Embeddings

1. **Exact Keyword and Acronym Mismatches**: Dense embeddings can over-generalize and miss exact numerical values, model serial numbers, or rare acronyms where lexical search excels.
2. **Asymmetric Search Queries**: Short search queries ("RAG flaws") have very different length and grammar compared to long descriptive paragraphs, requiring specialized asymmetric embedding models or task-type prompts.
3. **Fixed Dimensionality Bottleneck**: Compressing complex paragraphs into a single vector inevitably discards nuanced sub-clauses.
