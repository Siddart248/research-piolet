"""API integration tests for ResearchPilot FastAPI endpoints.

Verifies:
- GET  /health
- POST /research
- POST /chat (including multi-turn memory retention)
- GET  /evaluation
"""

import sys
from pathlib import Path
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_endpoint():
    """Verify POST /research executes agent workflow and returns ResearchResponse."""
    payload = {"query": "What are the limitations of RAG?"}
    response = client.post("/research", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert "key_points" in data
    assert "sources" in data
    assert "confidence" in data
    assert "insufficient_context" in data
    assert "tool_calls" in data
    assert data["insufficient_context"] is False
    assert "rag.md" in data["sources"]


def test_chat_endpoint_with_memory():
    """Verify POST /chat preserves conversation context across turns."""
    session_id = "test_memory_thread_01"

    # Turn 1: Math question
    res1 = client.post("/chat", json={"message": "What is 17% of 8500?", "session_id": session_id})
    assert res1.status_code == 200
    data1 = res1.json()
    assert "1445" in data1["answer"]

    # Turn 2: Follow-up referencing turn 1
    res2 = client.post(
        "/chat",
        json={"message": "Can you summarize what you just calculated?", "session_id": session_id},
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert "1445" in data2["answer"] or "8500" in data2["answer"]


def test_evaluation_endpoint():
    """Verify GET /evaluation returns latest summary metrics."""
    response = client.get("/evaluation")
    assert response.status_code == 200

    data = response.json()
    assert "metrics" in data
    assert "overall_pass_rate" in data["metrics"]
    assert "failure_taxonomy_breakdown" in data
