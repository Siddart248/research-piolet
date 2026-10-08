"""Tool definitions and safe execution engine for ResearchPilot.

Provides:
- calculate: Safe AST-based arithmetic evaluator (no direct eval()).
- retrieve_knowledge: Retrieval tool interfacing with the local knowledge base.
- tool registry and dispatcher.
"""

import ast
import logging
import operator
import re
from typing import Any

from app.retrieval import LocalVectorStore
from app.schemas import CalculatorOutput

logger = logging.getLogger("tools")

# Supported safe arithmetic operators
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast_node(node: ast.AST) -> float | int:
    """Recursively evaluate only safe mathematical AST nodes."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Non-numeric literal: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division by zero is undefined.")
        return SAFE_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_ast_node(node.operand)
        return SAFE_OPERATORS[op_type](operand)

    raise ValueError(f"Unsupported AST node expression: {type(node).__name__}")


def calculate(expression: str) -> CalculatorOutput:
    """Safely evaluate basic arithmetic expressions without arbitrary eval().

    Handles standard operations (+, -, *, /, **, //, %) and common percentage patterns
    such as '17% of 8500' or '15% * 2400'.

    Args:
        expression: Mathematical text string, e.g. '17% of 8500' or '1250 * 4 + 300'.

    Returns:
        CalculatorOutput: Result of computation or error message.
    """
    logger.info(f"[TOOL] Calculator called with expression: {expression!r}")

    if not expression or not expression.strip():
        return CalculatorOutput(
            expression=expression,
            result=None,
            formatted_result="Error: Empty expression.",
            success=False,
            error="Empty expression",
        )

    try:
        # Preprocess expression: remove commas in numbers, normalize carets
        cleaned = expression.strip().replace(",", "")
        cleaned = cleaned.replace("^", "**")

        # Normalize percentage phrases: '17% of 8500' -> '(17 / 100.0) * 8500'
        cleaned = re.sub(
            r"(\d+(?:\.\d+)?)\s*%\s*(?:of|\*)\s*(\d+(?:\.\d+)?)",
            r"(\1 / 100.0) * \2",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Normalize standalone percentage: '15%' -> '(15 / 100.0)'
        cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1 / 100.0)", cleaned)

        # Parse AST in 'eval' mode (disallows statements, assignments, imports)
        tree = ast.parse(cleaned, mode="eval")
        raw_result = _eval_ast_node(tree.body)

        # Round floats if near integers
        if isinstance(raw_result, float) and raw_result.is_integer():
            final_result = int(raw_result)
        elif isinstance(raw_result, float):
            final_result = round(raw_result, 6)
        else:
            final_result = raw_result

        formatted = str(final_result)
        logger.info(f"[TOOL] Calculator result: {expression} = {formatted}")

        return CalculatorOutput(
            expression=expression,
            result=final_result,
            formatted_result=formatted,
            success=True,
        )

    except ZeroDivisionError as err:
        logger.warning(f"[TOOL] Calculator error: {err}")
        return CalculatorOutput(
            expression=expression,
            result=None,
            formatted_result=f"Error: {err}",
            success=False,
            error=str(err),
        )
    except Exception as err:
        logger.warning(f"[TOOL] Calculator failed on expression {expression!r}: {err}")
        return CalculatorOutput(
            expression=expression,
            result=None,
            formatted_result=f"Error: Invalid arithmetic expression '{expression}'. {err}",
            success=False,
            error=str(err),
        )


def retrieve_knowledge(
    query: str,
    top_k: int = 3,
    threshold: float | None = None,
    store: LocalVectorStore | None = None,
) -> dict[str, Any]:
    """Retrieve relevant passages from the local knowledge base.

    Args:
        query: Research query text.
        top_k: Number of chunks to retrieve.
        threshold: Optional custom similarity threshold.
        store: Optional preloaded LocalVectorStore.

    Returns:
        dict: Summary of retrieval output including chunks, sources, and insufficient_context flag.
    """
    logger.info(f"[TOOL] Knowledge retrieval called for query: {query!r}")

    try:
        vector_store = store or LocalVectorStore.load()
    except FileNotFoundError:
        logger.warning("[TOOL] Vector store not found on disk. Building index...")
        from app.retrieval import build_index
        vector_store = build_index()

    retrieval_res = vector_store.retrieve(query=query, top_k=top_k, threshold=threshold)

    return {
        "query": retrieval_res.query,
        "best_similarity": retrieval_res.best_similarity,
        "threshold": retrieval_res.threshold,
        "insufficient_context": retrieval_res.insufficient_context,
        "sources": sorted(list({c.source for c in retrieval_res.chunks})),
        "chunks": [c.model_dump() for c in retrieval_res.chunks],
    }


# Tool Dispatcher
TOOL_REGISTRY = {
    "calculate": calculate,
    "retrieve_knowledge": retrieve_knowledge,
}
