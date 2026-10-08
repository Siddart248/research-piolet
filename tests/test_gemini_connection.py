"""Test script to verify Google Gemini API connectivity.

Loads GEMINI_API_KEY, instantiates the GenAI client, sends a test prompt,
and prints the model's response.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings, get_gemini_client


def test_gemini_connection() -> bool:
    """Test connecting to Gemini API and generating a simple response."""
    print("=" * 60)
    print("ResearchPilot: Gemini API Connectivity Test")
    print("=" * 60)

    try:
        settings = get_settings()
        print(f"[*] Configuration loaded successfully.")
        print(f"[*] Target Model : {settings.gemini_model}")
        print(f"[*] API Key Status: Configured (masked: {repr(settings)})")
    except ValueError as err:
        print(f"\n[ERROR] Configuration failed: {err}")
        return False

    try:
        print("\n[*] Initializing Gemini client...")
        client = get_gemini_client(settings)

        test_prompt = "Say 'Gemini connection successful!' and nothing else."
        print(f"[*] Sending prompt: {test_prompt!r}")

        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=test_prompt,
        )

        print("\n[SUCCESS] Response received from Gemini API:")
        print("-" * 40)
        print(response.text.strip())
        print("-" * 40)
        return True

    except Exception as err:
        print(f"\n[ERROR] Gemini API call failed: {type(err).__name__}: {err}")
        return False


if __name__ == "__main__":
    success = test_gemini_connection()
    sys.exit(0 if success else 1)
