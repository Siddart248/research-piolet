"""Embedding generation and vector operations for ResearchPilot.

Provides:
- embed_text: generates dense vector representation for a single text.
- embed_texts: batch generates embeddings for a list of texts.
- cosine_similarity: computes cosine similarity between vectors.
- Support for official Google Gemini API (text-embedding-004) with
  deterministic fallback for offline testing/verification when API keys are absent.
"""

import hashlib
import logging
import os
import re
from typing import Sequence
import numpy as np
from google import genai

from app.config import get_gemini_client, get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768


def embed_text_deterministic(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Generate a deterministic, normalized embedding vector from text using subword n-grams.

    Useful for offline testing, local unit tests, and verifying vector math
    without consuming external API quota or requiring network access.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return vec.tolist()

    for token in tokens:
        # Full token hash
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % dim
        vec[h] += 1.0
        # Character trigrams for morphological and subword capture
        for i in range(max(0, len(token) - 2)):
            sub = token[i : i + 3]
            h_sub = int(hashlib.sha256(sub.encode("utf-8")).hexdigest(), 16) % dim
            vec[h_sub] += 0.4

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.tolist()


def embed_text(
    text: str,
    client: genai.Client | None = None,
    model: str | None = None,
    use_mock: bool | None = None,
) -> list[float]:
    """Generate an embedding vector for a single text string using Google Gemini SDK.

    Args:
        text: Input string to embed.
        client: Optional pre-initialized GenAI Client.
        model: Optional embedding model name (defaults to Settings.embedding_model).
        use_mock: If True, forces deterministic local embedder without calling Gemini API.
                  If None, checks USE_MOCK_EMBEDDINGS env var or falls back if key is absent.

    Returns:
        list[float]: Embedding vector values.

    Raises:
        ValueError: If text is empty or API key is missing and use_mock is False.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    if use_mock is None:
        use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "").lower() in ("true", "1")

    if use_mock:
        return embed_text_deterministic(text)

    # Attempt to load settings and client
    settings = get_settings(require_api_key=False)
    target_model = model or settings.embedding_model

    if not settings.gemini_api_key or settings.gemini_api_key == "your_api_key_here":
        # API key is missing
        logger.warning(
            "GEMINI_API_KEY is not configured. Falling back to deterministic local embedding."
        )
        return embed_text_deterministic(text)

    if client is None:
        client = get_gemini_client(settings)

    response = client.models.embed_content(
        model=target_model,
        contents=text,
    )

    if not response.embeddings or not response.embeddings[0].values:
        raise ValueError(f"No embedding values returned from Gemini model {target_model}")

    return list(response.embeddings[0].values)


def embed_texts(
    texts: Sequence[str],
    client: genai.Client | None = None,
    model: str | None = None,
    use_mock: bool | None = None,
    batch_size: int = 16,
) -> list[list[float]]:
    """Batch generate embedding vectors for a sequence of texts.

    Args:
        texts: Sequence of strings to embed.
        client: Optional pre-initialized GenAI Client.
        model: Optional model name.
        use_mock: Force deterministic embedding.
        batch_size: Number of texts per batch request.

    Returns:
        list[list[float]]: List of embedding vectors.
    """
    if not texts:
        return []

    if use_mock is None:
        use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "").lower() in ("true", "1")

    if use_mock:
        return [embed_text_deterministic(t) for t in texts]

    settings = get_settings(require_api_key=False)
    target_model = model or settings.embedding_model

    if not settings.gemini_api_key or settings.gemini_api_key == "your_api_key_here":
        logger.warning(
            "GEMINI_API_KEY is not configured. Falling back to deterministic local embeddings."
        )
        return [embed_text_deterministic(t) for t in texts]

    if client is None:
        client = get_gemini_client(settings)

    all_embeddings: list[list[float]] = []

    # Process in batches to comply with API payload limits
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        response = client.models.embed_content(
            model=target_model,
            contents=batch,
        )
        if not response.embeddings:
            raise ValueError(f"No embeddings returned from Gemini model {target_model}")

        for emb in response.embeddings:
            all_embeddings.append(list(emb.values))

    return all_embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector/matrix a and vector b.

    Args:
        a: 1D array of shape (D,) or 2D array of shape (N, D).
        b: 1D array of shape (D,).

    Returns:
        np.ndarray: Cosine similarity scalar (if a is 1D) or 1D array of shape (N,) (if a is 2D).
    """
    b_norm = float(np.linalg.norm(b))
    if b_norm == 0.0:
        return np.zeros(a.shape[0] if a.ndim > 1 else 1, dtype=np.float32)

    b_unit = b / b_norm

    if a.ndim == 1:
        a_norm = float(np.linalg.norm(a))
        if a_norm == 0.0:
            return 0.0
        return float(np.dot(a / a_norm, b_unit))

    # 2D case: a is (N, D)
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    a_norms[a_norms == 0] = 1e-10
    a_normalized = a / a_norms

    similarities = np.dot(a_normalized, b_unit)
    return np.asarray(similarities, dtype=np.float32)
