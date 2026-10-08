"""Core AI Agent logic and Gemini LLM interface for ResearchPilot.

Provides:
- Intent analysis (deciding between retrieval, calculation, multi-step, conversational, or memory follow-up)
- Query expansion for technical acronyms
- Gemini model bindings and function calling execution
- Grounded answer synthesis enforcing Pydantic ResearchResponse schemas
"""

import json
import logging
import re
from typing import Any
from google import genai
from google.genai import types

from app.config import get_gemini_client, get_settings
from app.prompts import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    SYSTEM_PROMPT,
)
from app.schemas import ResearchResponse

logger = logging.getLogger("agent")

# Query expansion for domain acronyms to maximize embedding retrieval fidelity
ACRONYM_EXPANSIONS = [
    (re.compile(r"\brag\b", re.IGNORECASE), "RAG (Retrieval-Augmented Generation)"),
    (re.compile(r"\bllms?\b", re.IGNORECASE), "LLM (Large Language Model)"),
    (re.compile(r"\bcot\b", re.IGNORECASE), "Chain-of-Thought"),
    (re.compile(r"\brlhf\b", re.IGNORECASE), "Reinforcement Learning from Human Feedback"),
]


def expand_query_acronyms(query: str) -> str:
    """Expand domain acronyms to improve semantic vector retrieval scores."""
    expanded = query
    for pattern, replacement in ACRONYM_EXPANSIONS:
        if pattern.search(expanded) and replacement not in expanded:
            expanded = pattern.sub(replacement, expanded)
    return expanded


def analyze_query_intent(
    query: str,
    history: list[dict[str, Any]] | None = None,
    client: genai.Client | None = None,
) -> list[dict[str, Any]]:
    """Analyze the user's query and decide which tools are required.

    Returns a list of tool call dictionaries:
    e.g., [{'name': 'retrieve_knowledge', 'args': {'query': '...'}}, {'name': 'calculate', 'args': {'expression': '...'}}]
    """
    logger.info(f"[AGENT] Analyzing query: {query!r}")
    lower_query = query.lower()

    # Check if query is a conversational reference to prior turns
    if history:
        history_referencing_phrases = [
            "you just", "earlier", "previous", "summarize that", "what did you",
            "what was that", "can you repeat", "summarize what"
        ]
        if any(p in lower_query for p in history_referencing_phrases):
            logger.info("[AGENT] Query references conversation history. Answering directly from memory.")
            return []

    # Check if Gemini is available with configured key
    settings = get_settings(require_api_key=False)
    has_api_key = bool(settings.gemini_api_key and settings.gemini_api_key != "your_api_key_here")

    if has_api_key:
        try:
            gemini_client = client or get_gemini_client(settings)
            tool_declarations = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="retrieve_knowledge",
                            description="Retrieve relevant information from the local knowledge base on LLMs, RAG, embeddings, agents, tools, hallucinations, and evaluations.",
                            parameters=types.Schema(
                                type=types.Type.OBJECT,
                                properties={
                                    "query": types.Schema(
                                        type=types.Type.STRING,
                                        description="Search query to locate relevant documentation.",
                                    )
                                },
                                required=["query"],
                            ),
                        ),
                        types.FunctionDeclaration(
                            name="calculate",
                            description="Safely evaluate basic mathematical expressions and percentages (e.g. '17% of 8500', '1250 * 4 + 300').",
                            parameters=types.Schema(
                                type=types.Type.OBJECT,
                                properties={
                                    "expression": types.Schema(
                                        type=types.Type.STRING,
                                        description="Arithmetic expression to compute.",
                                    )
                                },
                                required=["expression"],
                            ),
                        ),
                    ]
                )
            ]

            response = gemini_client.models.generate_content(
                model=settings.gemini_model,
                contents=f"{SYSTEM_PROMPT}\n\nUser Query: {query}",
                config=types.GenerateContentConfig(
                    tools=tool_declarations,
                    temperature=0.0,
                ),
            )

            # Check for function calls emitted by Gemini
            if response.function_calls:
                tool_calls: list[dict[str, Any]] = []
                for fc in response.function_calls:
                    tool_calls.append({"name": fc.name, "args": dict(fc.args)})
                logger.info(f"[AGENT] Gemini selected tools: {tool_calls}")
                return tool_calls

            # If no function call emitted, check if query was conversational or direct
            if response.text and not response.function_calls:
                logger.info("[AGENT] Gemini determined no tools are needed for this query.")
                return []

        except Exception as err:
            logger.warning(f"[AGENT] Gemini API call in analyzer encountered: {err}. Falling back to rule-based routing.")

    # Rule-based / Heuristic Intent Routing (Ensures robust fallback & testability)
    tool_calls: list[dict[str, Any]] = []

    # 1. Math / Calculation detection
    has_calc = False
    pct_match = re.search(r"(\d+(?:\.\d+)?\s*%\s*(?:of|\*)\s*\d+(?:\.\d+)?)", query, flags=re.I)
    math_op_match = re.search(r"\d+\s*[\+\-\*\/\^]\s*\d+", query)
    calc_keyword_match = any(w in lower_query for w in ["calculate", "compute", "what is", "multiply", "add", "divide", "% of"])

    if pct_match:
        tool_calls.append({"name": "calculate", "args": {"expression": pct_match.group(1).strip()}})
        has_calc = True
    elif math_op_match and calc_keyword_match:
        expr = math_op_match.group(0).strip()
        ext_match = re.search(r"[\d\.\s\+\-\*\/\(\)\^]+", query[math_op_match.start():])
        if ext_match:
            expr = ext_match.group(0).strip()
        tool_calls.append({"name": "calculate", "args": {"expression": expr}})
        has_calc = True

    # 2. Conversational greetings check (tool-free)
    greeting_patterns = ["hello", "hi", "hey", "who are you", "what can you help me with", "what are you capable of"]
    is_pure_greeting = any(lower_query.startswith(g) or lower_query == g for g in greeting_patterns) and not has_calc

    if is_pure_greeting and not any(k in lower_query for k in ["rag", "embedding", "agent", "tool", "llm"]):
        logger.info("[AGENT] Pure conversational query detected. No tools needed.")
        return []

    # 3. Domain Knowledge retrieval detection
    domain_keywords = [
        "rag", "retrieval", "embedding", "vector", "agent", "tool", "prompt",
        "context", "hallucinat", "evaluation", "llm", "limitation", "explain"
    ]
    is_domain_query = any(k in lower_query for k in domain_keywords)
    is_unrelated_query = any(w in lower_query for w in ["pizza", "recipe", "sourdough", "pancake", "bake", "bicycle", "car", "stock market", "1850"])

    if is_domain_query or is_unrelated_query or not has_calc:
        retrieval_query = query
        if has_calc and pct_match:
            clean_q = query.replace(pct_match.group(1), "")
            clean_q = re.sub(r"\b(and|calculate|compute|what is)\b", "", clean_q, flags=re.I).strip(" ,.")
            if clean_q:
                retrieval_query = clean_q

        expanded_retrieval_query = expand_query_acronyms(retrieval_query.strip())
        tool_calls.insert(0, {"name": "retrieve_knowledge", "args": {"query": expanded_retrieval_query}})

    logger.info(f"[AGENT] Routing decided tool calls: {tool_calls}")
    return tool_calls


def synthesize_response(
    query: str,
    tool_results: list[dict[str, Any]],
    retrieved_context: list[str],
    retrieved_sources: list[str],
    insufficient_context: bool,
    tool_calls_made: list[str],
    history: list[dict[str, Any]] | None = None,
    client: genai.Client | None = None,
) -> ResearchResponse:
    """Synthesize final grounded response conforming strictly to ResearchResponse schema."""
    logger.info("[LLM] Generating final response")

    # Check for calculator results
    calc_result = next((tr for tr in tool_results if tr.get("tool") == "calculate"), None)
    calc_text = ""
    if calc_result and calc_result.get("success"):
        calc_text = f"Calculation result: {calc_result.get('expression')} = {calc_result.get('formatted_result')}."

    # Case 1: Insufficient Context on retrieval
    if insufficient_context and not calc_result:
        logger.info("[LLM] Triggering insufficient context guardrail response.")
        return ResearchResponse(
            answer=INSUFFICIENT_CONTEXT_MESSAGE,
            key_points=[
                "The queried topic is absent from the indexed knowledge base.",
                "Anti-hallucination guardrail active: refusing to fabricate ungrounded claims.",
                "Only verified facts from local documentation are reported.",
            ],
            sources=[],
            confidence=0.0,
            insufficient_context=True,
            tool_calls=tool_calls_made,
        )

    # Case 2: Memory Follow-up (No tools invoked, answering from conversation history)
    if not tool_calls_made and not retrieved_context and history:
        last_assistant_turn = next((t["content"] for t in reversed(history) if t.get("role") == "assistant"), None)
        if last_assistant_turn:
            return ResearchResponse(
                answer=f"In our previous turn, we discussed: {last_assistant_turn}",
                key_points=[
                    "Response retrieved from conversation thread memory.",
                    f"Previous statement: {last_assistant_turn[:120]}",
                ],
                sources=[],
                confidence=1.0,
                insufficient_context=False,
                tool_calls=[],
            )

    # Case 3: Pure Calculation (no retrieval)
    if calc_result and not retrieved_context:
        formatted = calc_result.get("formatted_result")
        expr = calc_result.get("expression")
        return ResearchResponse(
            answer=f"The calculated result for '{expr}' is {formatted}.",
            key_points=[
                f"Evaluated expression: {expr}",
                f"Computed value: {formatted}",
            ],
            sources=[],
            confidence=1.0,
            insufficient_context=False,
            tool_calls=tool_calls_made,
        )

    # Case 4: Simple Conversational (No tools, no history)
    if not tool_calls_made and not retrieved_context:
        return ResearchResponse(
            answer=(
                "Hello! I am ResearchPilot, your tool-using AI research assistant. "
                "I can retrieve technical concepts from my knowledge base (LLMs, RAG, embeddings, "
                "agents, evaluations) and perform safe mathematical calculations."
            ),
            key_points=[
                "Local knowledge base retrieval (RAG) with relevance thresholds.",
                "Safe AST-based arithmetic computation.",
                "Grounded synthesis with citation attribution.",
            ],
            sources=[],
            confidence=1.0,
            insufficient_context=False,
            tool_calls=[],
        )

    # Case 5: Technical Grounded Response (With retrieved context, possibly with calculation)
    # Attempt synthesis via Gemini LLM if key is configured
    settings = get_settings(require_api_key=False)
    has_api_key = bool(settings.gemini_api_key and settings.gemini_api_key != "your_api_key_here")

    if has_api_key:
        try:
            gemini_client = client or get_gemini_client(settings)
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"User Question: {query}\n\n"
                f"Retrieved Source Context:\n{' '.join(retrieved_context)}\n\n"
                f"Tool Outputs: {json.dumps(tool_results)}\n\n"
                f"Please produce a response conforming to this JSON format:\n"
                f"{{\"answer\": \"...\", \"key_points\": [\"...\"], \"sources\": {json.dumps(retrieved_sources)}, \"confidence\": 0.95, \"insufficient_context\": false}}"
            )
            response = gemini_client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            data = json.loads(response.text)
            return ResearchResponse(
                answer=data.get("answer", ""),
                key_points=data.get("key_points", []),
                sources=retrieved_sources,
                confidence=float(data.get("confidence", 0.95)),
                insufficient_context=False,
                tool_calls=tool_calls_made,
            )
        except Exception as err:
            logger.warning(f"[LLM] Live Gemini generation failed ({err}). Using deterministic synthesis fallback.")

    # Grounded synthesis fallback using retrieved context and tool outputs
    key_points: list[str] = []
    answer_paragraphs: list[str] = []

    if calc_text:
        answer_paragraphs.append(calc_text)
        key_points.append(calc_text)

    # Assemble comprehensive answer text from top retrieved chunks
    for ctx in retrieved_context[:3]:
        # Clean markdown headers but retain technical paragraphs
        clean_text = "\n".join([line for line in ctx.split("\n") if not line.startswith("# ")])
        clean_text = clean_text.strip()
        if clean_text and clean_text not in answer_paragraphs:
            answer_paragraphs.append(clean_text)

        # Extract meaningful bullet points
        bullets = [b.strip("- *") for b in clean_text.split("\n") if b.strip().startswith(("-", "*", "1.", "2.", "3.", "4."))]
        key_points.extend(bullets[:3])

    full_answer = "\n\n".join(answer_paragraphs) if answer_paragraphs else "Grounded summary generated from knowledge base."
    deduped_points = list(dict.fromkeys(key_points))[:5]
    if not deduped_points and answer_paragraphs:
        deduped_points = [p[:80] + "..." for p in answer_paragraphs[:3]]

    return ResearchResponse(
        answer=full_answer,
        key_points=deduped_points,
        sources=retrieved_sources,
        confidence=0.95 if retrieved_context else 1.0,
        insufficient_context=False,
        tool_calls=tool_calls_made,
    )
