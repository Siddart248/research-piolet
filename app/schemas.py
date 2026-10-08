"""Data schemas for ResearchPilot.

Defines Pydantic models for knowledge base documents, chunks, retrieval results,
tool inputs/outputs, structured research responses, API request payloads, and LangGraph agent state.
"""

from typing import Any, TypedDict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Knowledge Base & Retrieval Schemas
# ---------------------------------------------------------------------------

class Document(BaseModel):
    """Raw document loaded from the knowledge base."""

    id: str = Field(description="Unique identifier, typically the filename.")
    content: str = Field(description="Full text content of the document.")
    source: str = Field(description="Filename or path relative to knowledge base directory.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional document attributes.")


class Chunk(BaseModel):
    """Chunk of text extracted from a Document with associated metadata."""

    chunk_id: str = Field(description="Unique ID for this chunk (e.g. filename_chunk_000).")
    source: str = Field(description="Source document name.")
    text: str = Field(description="Chunk textual content.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk attributes (e.g., char offsets).")


class RetrievedChunk(BaseModel):
    """Retrieved chunk enriched with cosine similarity score."""

    chunk_id: str
    source: str
    text: str
    similarity: float = Field(description="Cosine similarity score against the query.")


class RetrievalResult(BaseModel):
    """Aggregated retrieval output for a query."""

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    best_similarity: float = Field(default=0.0)
    threshold: float = Field(description="Minimum similarity score required to accept context.")
    insufficient_context: bool = Field(
        default=False,
        description="True if best_similarity is strictly below threshold or no chunks were found.",
    )


# ---------------------------------------------------------------------------
# Tool Input / Output Schemas
# ---------------------------------------------------------------------------

class CalculatorInput(BaseModel):
    """Input payload for the safe calculator tool."""

    expression: str = Field(
        description="Arithmetic expression to safely evaluate, e.g. '17% of 8500' or '1250 * 4 + 300'."
    )


class CalculatorOutput(BaseModel):
    """Output payload from the safe calculator tool."""

    expression: str
    result: float | int | None = None
    formatted_result: str = ""
    success: bool = True
    error: str | None = None


class RetrievalToolInput(BaseModel):
    """Input payload for the knowledge retrieval tool."""

    query: str = Field(description="Search query to locate relevant knowledge base documents.")
    top_k: int = Field(default=3, description="Number of top chunks to retrieve.")


# ---------------------------------------------------------------------------
# Structured Agent Response Schema
# ---------------------------------------------------------------------------

class ResearchResponse(BaseModel):
    """Guaranteed structured output contract for ResearchPilot research answers."""

    answer: str = Field(description="Comprehensive, factual response grounded in retrieved context or calculations.")
    key_points: list[str] = Field(
        default_factory=list,
        description="Bullet-point summary of core insights.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of cited source filenames (e.g. 'rag.md'). Empty if answer is ungrounded or conversational.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 reflecting factual grounding certainty.",
    )
    insufficient_context: bool = Field(
        default=False,
        description="True if knowledge base lacked necessary information to answer truthfully.",
    )
    tool_calls: list[str] = Field(
        default_factory=list,
        description="List of tools invoked during query execution.",
    )


# ---------------------------------------------------------------------------
# API Request / Response Payloads
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    """Request payload for POST /research."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language research query.",
        examples=["What are the limitations of RAG?"],
    )


class ChatRequest(BaseModel):
    """Request payload for POST /chat."""

    message: str = Field(
        ...,
        min_length=1,
        description="User message in a conversational thread.",
        examples=["What is 15% of 2400?"],
    )
    session_id: str = Field(
        default="default",
        description="Thread or session identifier for multi-turn conversational memory.",
    )


# ---------------------------------------------------------------------------
# LangGraph Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Typed state dictionary managed across LangGraph workflow nodes."""

    user_query: str
    conversation_history: list[dict[str, Any]]
    retrieved_context: list[str]
    retrieved_sources: list[str]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    final_answer: ResearchResponse | None
    insufficient_context: bool
    metadata: dict[str, Any]
