# Hallucinations in Large Language Models

A hallucination occurs when an LLM generates statements that are factually false, unfaithful to provided source material, or entirely fabricated, often expressed with unjustified confidence.

## Categories of Hallucination

1. **Intrinsic Hallucination**: The model's response directly contradicts facts explicitly stated in the provided prompt or context documents.
2. **Extrinsic Hallucination**: The model introduces external claims that cannot be verified or substantiated from the provided context (even if those claims might coincidentally be true in the real world).

## Root Causes of Hallucinatory Behavior

- **Autoregressive Pressure**: LLMs are trained to maximize sequence likelihood. When facing an unfamiliar topic or ambiguous context, continuing the token stream fluently takes precedence over asserting ignorance.
- **Parametric Prior Interference**: When retrieved external context conflicts with strong parametric associations memorized during pre-training, the model frequently defaults back to its pre-trained prior.
- **Partial or Low-Relevance Context**: Providing fragments that hint at an answer without fully explaining it invites the model to guess or fill in the blanks.

## Engineering Defenses Against Hallucination

- **Strict Relevance Thresholds**: Halting synthesis and declaring `insufficient_context = True` if similarity scores between query and knowledge base fail to pass a rigorous threshold.
- **Constrained Decoding and System Instructions**: Explicitly instructing the model to reply "I do not have sufficient information in the provided context to answer this question" rather than guessing.
- **Post-Generation Verification**: Running an automated evaluation pass that compares each generated sentence against the source chunks to guarantee complete citation grounding.
