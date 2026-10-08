"""FastAPI application entrypoint for ResearchPilot.

Provides REST endpoints:
- GET  /health      : System health check.
- POST /research    : Research query processing returning structured ResearchResponse.
- POST /chat        : Stateful multi-turn conversational interaction with memory.
- GET  /evaluation  : Triggers or retrieves latest engineering evaluation summary.
"""

from contextlib import asynccontextmanager
import logging
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.evaluation import DEFAULT_SUMMARY_PATH, run_evaluation
from app.graph import run_research_agent
from app.logging_config import setup_logging
from app.memory import memory_store
from app.schemas import ChatRequest, ResearchRequest, ResearchResponse

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    setup_logging("INFO")
    logger.info("ResearchPilot API initialized and ready.")
    yield
    logger.info("ResearchPilot API shut down.")


app = FastAPI(
    title="ResearchPilot API",
    description=(
        "Tool-using AI research agent demonstrating local RAG, safe AST tool calling, "
        "LangGraph stateful workflows, conversation memory, and structured outputs."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for developer usability
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Health check endpoint to verify service availability."""
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse, tags=["Research Agent"])
def submit_research_query(payload: ResearchRequest) -> ResearchResponse:
    """Execute the full ResearchPilot agent on a standalone research query.

    Analyzes intent, executes retrieval and/or tools, and generates a grounded response.
    """
    logger.info(f"[API] Received /research query: {payload.query!r}")
    try:
        response: ResearchResponse = run_research_agent(query=payload.query)
        return response
    except Exception as err:
        logger.error(f"[API] Error processing /research query: {err}")
        raise HTTPException(status_code=500, detail=f"Agent workflow error: {str(err)}")


@app.post("/chat", response_model=ResearchResponse, tags=["Conversational Agent"])
def submit_chat_message(payload: ChatRequest) -> ResearchResponse:
    """Submit a message in a stateful conversation thread with sliding-window memory.

    Preserves recent conversational context and updates thread state.
    """
    session_id = payload.session_id or "default"
    logger.info(f"[API] Received /chat message in session '{session_id}': {payload.message!r}")

    try:
        # Retrieve recent history for this session
        history = memory_store.get_history_dicts(session_id)

        # Run agent with conversation context
        response: ResearchResponse = run_research_agent(query=payload.message, history=history)

        # Record turns into sliding-window memory
        memory_store.add_turn(session_id=session_id, role="user", content=payload.message)
        memory_store.add_turn(
            session_id=session_id,
            role="assistant",
            content=response.answer,
            metadata={"confidence": response.confidence, "sources": response.sources},
        )

        return response
    except Exception as err:
        logger.error(f"[API] Error in /chat session '{session_id}': {err}")
        raise HTTPException(status_code=500, detail=f"Chat agent error: {str(err)}")


@app.get("/evaluation", tags=["Evaluation"])
def get_evaluation_summary(run_fresh: bool = Query(default=False, description="Run fresh evaluation if True")) -> dict[str, Any]:
    """Retrieve the latest engineering evaluation summary or trigger a fresh run."""
    import json

    if run_fresh or not DEFAULT_SUMMARY_PATH.exists():
        logger.info("[API] Running fresh evaluation suite on request...")
        summary = run_evaluation()
        return summary

    try:
        summary = json.loads(DEFAULT_SUMMARY_PATH.read_text(encoding="utf-8"))
        return summary
    except Exception as err:
        logger.error(f"[API] Failed to read evaluation summary: {err}")
        summary = run_evaluation()
        return summary
