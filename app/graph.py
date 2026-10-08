"""LangGraph stateful workflow implementation for ResearchPilot AI Agent.

Defines:
- StateGraph with nodes: analyze_query, retrieve_context, execute_tools, generate_answer
- Conditional routing edges based on intent classification and retrieval adequacy
- run_research_agent function for single-call agent execution
"""

import logging
from typing import Any
from langgraph.graph import END, START, StateGraph

from app.agent import analyze_query_intent, synthesize_response
from app.schemas import AgentState, ResearchResponse
from app.tools import calculate, retrieve_knowledge

logger = logging.getLogger("graph")


def analyze_query(state: AgentState) -> dict[str, Any]:
    """Node 1: Inspect query and identify necessary tool calls."""
    query = state.get("user_query", "")
    history = state.get("conversation_history", [])
    tool_calls = analyze_query_intent(query, history=history)
    return {
        "tool_calls": tool_calls,
        "metadata": {"analysis_complete": True},
    }


def route_after_analysis(state: AgentState) -> str:
    """Conditional router determining next step after query analysis."""
    tool_calls = state.get("tool_calls", [])
    call_names = [call.get("name") for call in tool_calls]

    if "retrieve_knowledge" in call_names:
        return "retrieve_context"
    elif "calculate" in call_names:
        return "execute_tools"
    else:
        return "generate_answer"


def retrieve_context(state: AgentState) -> dict[str, Any]:
    """Node 2: Execute knowledge retrieval against the local knowledge base."""
    tool_calls = state.get("tool_calls", [])
    retrieval_call = next((c for c in tool_calls if c.get("name") == "retrieve_knowledge"), None)
    
    query = retrieval_call["args"]["query"] if retrieval_call and "args" in retrieval_call else state.get("user_query", "")
    logger.info(f"[RETRIEVAL] Searching knowledge base for: {query!r}")

    result = retrieve_knowledge(query=query)
    is_insufficient = result.get("insufficient_context", False)

    chunks = result.get("chunks", [])
    sources = result.get("sources", [])
    retrieved_texts = [c.get("text", "") for c in chunks]

    existing_results = list(state.get("tool_results", []))
    existing_results.append({
        "tool": "retrieve_knowledge",
        "query": query,
        "best_similarity": result.get("best_similarity", 0.0),
        "insufficient_context": is_insufficient,
        "sources": sources,
    })

    return {
        "retrieved_context": retrieved_texts,
        "retrieved_sources": sources,
        "tool_results": existing_results,
        "insufficient_context": is_insufficient,
    }


def route_after_retrieval(state: AgentState) -> str:
    """Conditional router determining next step after retrieval."""
    # If context is insufficient, immediately route to generate answer to refuse hallucination
    if state.get("insufficient_context", False):
        return "generate_answer"

    # Check if calculation is also requested in a multi-step query
    tool_calls = state.get("tool_calls", [])
    call_names = [c.get("name") for c in tool_calls]
    if "calculate" in call_names:
        return "execute_tools"

    return "generate_answer"


def execute_tools(state: AgentState) -> dict[str, Any]:
    """Node 3: Safely execute calculator or deterministic helper tools."""
    tool_calls = state.get("tool_calls", [])
    existing_results = list(state.get("tool_results", []))

    for call in tool_calls:
        name = call.get("name")
        args = call.get("args", {})

        if name == "calculate":
            expression = args.get("expression", "")
            logger.info(f"[TOOL] Calculator called for expression: {expression!r}")
            output = calculate(expression)
            existing_results.append({
                "tool": "calculate",
                "expression": output.expression,
                "result": output.result,
                "formatted_result": output.formatted_result,
                "success": output.success,
                "error": output.error,
            })

    return {"tool_results": existing_results}


def generate_answer(state: AgentState) -> dict[str, Any]:
    """Node 4: Synthesize structured, grounded final ResearchResponse."""
    tool_calls = state.get("tool_calls", [])
    tool_calls_made = [c.get("name") for c in tool_calls]

    response = synthesize_response(
        query=state.get("user_query", ""),
        tool_results=state.get("tool_results", []),
        retrieved_context=state.get("retrieved_context", []),
        retrieved_sources=state.get("retrieved_sources", []),
        insufficient_context=state.get("insufficient_context", False),
        tool_calls_made=tool_calls_made,
        history=state.get("conversation_history", []),
    )

    return {"final_answer": response}


def build_research_agent_graph():
    """Build and compile the LangGraph workflow."""
    builder = StateGraph(AgentState)

    # Register workflow nodes
    builder.add_node("analyze_query", analyze_query)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("generate_answer", generate_answer)

    # Register workflow edges
    builder.add_edge(START, "analyze_query")
    builder.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "retrieve_context": "retrieve_context",
            "execute_tools": "execute_tools",
            "generate_answer": "generate_answer",
        },
    )
    builder.add_conditional_edges(
        "retrieve_context",
        route_after_retrieval,
        {
            "execute_tools": "execute_tools",
            "generate_answer": "generate_answer",
        },
    )
    builder.add_edge("execute_tools", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()


# Compiled Singleton Graph
research_agent_graph = build_research_agent_graph()


def run_research_agent(query: str, history: list[dict[str, Any]] | None = None) -> ResearchResponse:
    """Execute the full ResearchPilot agent workflow for a given user query.

    Args:
        query: User research question or command.
        history: Optional conversation message history.

    Returns:
        ResearchResponse: Strongly typed, grounded research response.
    """
    initial_state: AgentState = {
        "user_query": query,
        "conversation_history": history or [],
        "retrieved_context": [],
        "retrieved_sources": [],
        "tool_calls": [],
        "tool_results": [],
        "final_answer": None,
        "insufficient_context": False,
        "metadata": {},
    }

    final_state = research_agent_graph.invoke(initial_state)
    return final_state["final_answer"]
