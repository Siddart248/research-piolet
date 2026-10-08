# Large Language Model Fundamentals

Large Language Models (LLMs) are deep learning architectures based on the Transformer architecture, specifically autoregressive decoder-only models such as GPT, Claude, and Gemini. These models are trained on massive text corpora to predict the next token in a sequence given preceding context tokens.

## Architecture and Pre-training

At the core of modern LLMs is the multi-head self-attention mechanism. Self-attention enables the model to weigh the significance of different tokens across the entire input sequence regardless of positional distance. During pre-training, the model optimizes cross-entropy loss over billions or trillions of tokens, learning linguistic patterns, factual associations, reasoning heuristics, and code syntax.

## Alignment: SFT and RLHF

Raw pre-trained foundation models are typically text completion engines. To transform them into useful conversational assistants, alignment pipelines are applied:
1. **Supervised Fine-Tuning (SFT)**: Training on curated demonstration dialogues where instructions are paired with high-quality assistant responses.
2. **Reinforcement Learning from Human Feedback (RLHF)** or Direct Preference Optimization (DPO): Aligning model behaviors with human intentions regarding helpfulness, harmlessness, and honesty.

## Context Windows and Tokenization

LLMs operate on discrete tokens rather than characters or words. Subword tokenizers (like Byte-Pair Encoding) map text into numerical token IDs. Every model has a defined maximum context length (the context window), bounding the number of input tokens plus output tokens processed in a single forward pass. While context windows have expanded from 4,000 tokens to millions of tokens, models still exhibit degradation in long-context retrieval (the "needle in a haystack" phenomenon and "lost in the middle" effect).

## Inference Characteristics and Non-determinism

LLM inference computes probability distributions over the vocabulary for each subsequent token. Sampling strategies such as temperature, top-p (nucleus sampling), and top-k govern token selection. Lower temperature values (e.g., 0.0 or 0.2) produce focused, deterministic outputs, whereas higher values (e.g., 0.7 to 1.0) encourage diversity and creative variation.
