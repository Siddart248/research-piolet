"""Configuration and Gemini Client initialization module for ResearchPilot.

Provides validated environment settings and the official Google Gemini SDK client.
Ensures API keys are never leaked, logged, or hardcoded.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Resolve project root directory and load .env if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_RETRIEVAL_THRESHOLD = 0.45
DEFAULT_RETRIEVAL_TOP_K = 5


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    gemini_api_key: str
    gemini_model: str = DEFAULT_GEMINI_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    retrieval_threshold: float = DEFAULT_RETRIEVAL_THRESHOLD
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K

    def __repr__(self) -> str:
        """Mask sensitive keys to prevent accidental leaks in logs or traces."""
        if not self.gemini_api_key:
            masked = "<empty>"
        elif len(self.gemini_api_key) > 8:
            masked = f"{self.gemini_api_key[:3]}...{self.gemini_api_key[-3:]}"
        else:
            masked = "***"
        return (
            f"Settings(gemini_model='{self.gemini_model}', "
            f"embedding_model='{self.embedding_model}', "
            f"retrieval_threshold={self.retrieval_threshold}, "
            f"gemini_api_key='{masked}')"
        )


def get_settings(require_api_key: bool = True) -> Settings:
    """Load and validate application settings from environment variables.

    Args:
        require_api_key: If True, raises ValueError if GEMINI_API_KEY is missing or a placeholder.

    Raises:
        ValueError: If require_api_key is True and GEMINI_API_KEY is not set or contains placeholder text.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if require_api_key:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please create a '.env' file in the project root "
                "(copy from .env.example) and set GEMINI_API_KEY with your valid API key."
            )

        if api_key == "your_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is still set to placeholder 'your_api_key_here'. "
                "Please replace it with a valid Google Gemini API key in your '.env' file."
            )

    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    emb_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL

    raw_threshold = os.getenv("RETRIEVAL_THRESHOLD", str(DEFAULT_RETRIEVAL_THRESHOLD)).strip()
    try:
        threshold = float(raw_threshold)
    except ValueError:
        threshold = DEFAULT_RETRIEVAL_THRESHOLD

    raw_top_k = os.getenv("RETRIEVAL_TOP_K", str(DEFAULT_RETRIEVAL_TOP_K)).strip()
    try:
        top_k = int(raw_top_k)
    except ValueError:
        top_k = DEFAULT_RETRIEVAL_TOP_K

    return Settings(
        gemini_api_key=api_key,
        gemini_model=model,
        embedding_model=emb_model,
        retrieval_threshold=threshold,
        retrieval_top_k=top_k,
    )


def get_gemini_client(settings: Settings | None = None) -> genai.Client:
    """Initialize and return the official Google GenAI Client.

    Args:
        settings: Optional preloaded Settings instance. If omitted, loads via get_settings().

    Returns:
        genai.Client: Configured Gemini client instance.
    """
    if settings is None:
        settings = get_settings(require_api_key=True)

    return genai.Client(api_key=settings.gemini_api_key)
