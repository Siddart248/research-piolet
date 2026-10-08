# Context Engineering and Window Management

Context engineering is the systematic discipline of curating, structuring, ordering, and filtering information injected into an LLM's context window to optimize retrieval accuracy, eliminate noise, and prevent reasoning degradation.

## Context Window Dynamics

While modern foundation models boast context windows exceeding 1 million tokens, effective context capacity is substantially narrower. Empirical research highlights two pervasive biases:
1. **Primacy Bias**: Disproportionately weighting instructions and examples positioned at the very beginning of the prompt.
2. **Recency Bias**: Disproportionately weighting text situated immediately prior to the generation point.
3. **Lost in the Middle**: Information located centrally within long context blocks suffers the highest retrieval failure and omission rates.

## Key Context Engineering Techniques

- **Explicit Delimiting & Source Tagging**: Enclosing each retrieved chunk within distinct XML or markdown blocks containing unambiguous metadata (e.g., `<document id="doc_1" source="rag.md">...</document>`).
- **Context Budgets and Token Quotas**: Setting hard limits on tokens allocated to system instructions, conversation history, retrieved documents, and tool outputs to prevent context overflow.
- **Deduplication and Relevance Ranking**: Filtering redundant or low-similarity chunks before injection to maximize the density of informative tokens.
- **Strict Grounding Directives**: Instructing the model to rely solely on the facts enclosed within context boundaries, treating external prior knowledge as invalid for domain questions.
