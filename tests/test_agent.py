"""Automated test suite for LangGraph ResearchPilot AI Agent.

Verifies the 5 core test cases required:
1. RAG question: 'What are the limitations of RAG?' -> retrieval is used.
2. Calculator question: 'What is 17% of 8500?' -> calculator tool is used, computes 1445.
3. Tool-free/simple question: 'Hello, what can you help me with?' -> no unnecessary tools used.
4. Unknown question: 'What is the secret recipe for authentic Neapolitan pizza dough?' -> insufficient_context is True, no hallucination.
5. Multi-step question: 'Explain RAG and calculate 15% of 2000.' -> retrieval + calculator + synthesis.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.graph import run_research_agent
from app.schemas import ResearchResponse


def test_rag_question():
    """Test 1: RAG question triggers knowledge retrieval tool."""
    query = "What are the limitations of RAG?"
    print(f"\n[TEST 1] RAG Question: {query!r}")
    res: ResearchResponse = run_research_agent(query)

    assert isinstance(res, ResearchResponse), "Result must be a ResearchResponse instance"
    assert "retrieve_knowledge" in res.tool_calls, f"Expected retrieve_knowledge in {res.tool_calls}"
    assert "calculate" not in res.tool_calls, "Calculator should not be invoked for pure RAG query"
    assert not res.insufficient_context, "RAG query should find sufficient context"
    assert "rag.md" in res.sources, f"Expected rag.md in {res.sources}"
    assert len(res.answer) > 20, "Answer should be substantive"
    print(f"[PASS] Retrieval tool used, cited: {res.sources}, confidence: {res.confidence}")


def test_calculator_question():
    """Test 2: Math question triggers safe calculator tool."""
    query = "What is 17% of 8500?"
    print(f"\n[TEST 2] Calculator Question: {query!r}")
    res: ResearchResponse = run_research_agent(query)

    assert isinstance(res, ResearchResponse), "Result must be a ResearchResponse instance"
    assert "calculate" in res.tool_calls, f"Expected calculate in {res.tool_calls}"
    assert "retrieve_knowledge" not in res.tool_calls, "Retrieval should not be invoked for pure math"
    assert "1445" in res.answer or any("1445" in pt for pt in res.key_points), (
        f"Answer or key points must contain 1445. Got: {res.answer}"
    )
    assert not res.insufficient_context
    print(f"[PASS] Calculator tool used, result 1445 contained in answer.")


def test_simple_conversational_question():
    """Test 3: Greeting / overview question answered directly without tools."""
    query = "Hello, what can you help me with?"
    print(f"\n[TEST 3] Conversational Question: {query!r}")
    res: ResearchResponse = run_research_agent(query)

    assert isinstance(res, ResearchResponse), "Result must be a ResearchResponse instance"
    assert len(res.tool_calls) == 0, f"Expected no tools for greeting, got {res.tool_calls}"
    assert not res.insufficient_context
    assert len(res.answer) > 10
    print(f"[PASS] Handled directly without unnecessary tool calls: {res.tool_calls}")


def test_unknown_question_anti_hallucination():
    """Test 4: Unrelated query triggers insufficient context and safe refusal."""
    query = "What is the secret recipe for authentic Neapolitan pizza dough?"
    print(f"\n[TEST 4] Unknown Query: {query!r}")
    res: ResearchResponse = run_research_agent(query)

    assert isinstance(res, ResearchResponse), "Result must be a ResearchResponse instance"
    assert res.insufficient_context is True, "Expected insufficient_context=True for unknown domain query"
    assert res.confidence == 0.0, "Expected confidence 0.0 on missing context"
    assert len(res.sources) == 0, "No sources should be cited for missing context"
    assert "does not contain sufficient information" in res.answer.lower() or "not contain enough" in res.answer.lower(), (
        f"Expected refusal message, got: {res.answer}"
    )
    print(f"[PASS] Successfully rejected unknown query with insufficient_context=True (No hallucination).")


def test_multistep_question():
    """Test 5: Multi-step query invokes both retrieval and calculator."""
    query = "Explain RAG and calculate 15% of 2000."
    print(f"\n[TEST 5] Multi-Step Question: {query!r}")
    res: ResearchResponse = run_research_agent(query)

    assert isinstance(res, ResearchResponse), "Result must be a ResearchResponse instance"
    assert "retrieve_knowledge" in res.tool_calls, f"Expected retrieve_knowledge in {res.tool_calls}"
    assert "calculate" in res.tool_calls, f"Expected calculate in {res.tool_calls}"
    assert not res.insufficient_context
    assert "300" in res.answer or any("300" in pt for pt in res.key_points), (
        f"Answer or key points must contain 300 (15% of 2000). Got: {res.answer}"
    )
    assert "rag.md" in res.sources, f"Expected rag.md in {res.sources}"
    print(f"[PASS] Multi-step query invoked both tools, computed 300, and cited rag.md.")


def run_all_agent_tests():
    print("=" * 60)
    print("ResearchPilot: LangGraph Agent Test Suite")
    print("=" * 60)

    test_rag_question()
    test_calculator_question()
    test_simple_conversational_question()
    test_unknown_question_anti_hallucination()
    test_multistep_question()

    print("\n" + "=" * 60)
    print("[ALL 5 AGENT TEST CASES PASSED SUCCESSFULLY]")
    print("=" * 60)


if __name__ == "__main__":
    run_all_agent_tests()
