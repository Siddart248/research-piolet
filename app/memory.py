"""Conversation memory management for ResearchPilot.

Provides:
- Session-based conversation message storage.
- Bounded sliding window memory to avoid context window explosion.
- Formatting helpers for prompt context engineering.
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any

logger = logging.getLogger("memory")


@dataclass
class ConversationTurn:
    """A single conversational turn."""

    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """In-memory bounded conversation history manager."""

    def __init__(self, max_turns: int = 8):
        """Initialize memory with maximum turns (user + assistant pairs) to retain per session."""
        self.max_turns = max_turns
        self._sessions: dict[str, list[ConversationTurn]] = {}

    def get_session(self, session_id: str = "default") -> list[ConversationTurn]:
        """Retrieve stored turns for a given session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return self._sessions[session_id]

    def add_turn(self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Append a conversational turn and prune history if exceeding max_turns."""
        session = self.get_session(session_id)
        session.append(ConversationTurn(role=role, content=content, metadata=metadata or {}))

        # Enforce sliding window to control prompt token budget
        if len(session) > self.max_turns * 2:
            self._sessions[session_id] = session[-(self.max_turns * 2) :]
            logger.debug(f"[MEMORY] Pruned session '{session_id}' to last {self.max_turns * 2} messages.")

    def get_history_dicts(self, session_id: str = "default") -> list[dict[str, Any]]:
        """Return history as standard list of dictionaries."""
        session = self.get_session(session_id)
        return [{"role": t.role, "content": t.content} for t in session]

    def format_for_context(self, session_id: str = "default", last_k: int = 4) -> str:
        """Format recent conversation history as clean text for prompt context."""
        session = self.get_session(session_id)
        recent = session[-last_k:] if len(session) > last_k else session
        if not recent:
            return "No previous conversation."

        formatted_lines = []
        for t in recent:
            prefix = "User" if t.role == "user" else "Assistant"
            # Truncate overly long assistant turns in history to prevent context pollution
            text = t.content[:200] + "..." if len(t.content) > 200 else t.content
            formatted_lines.append(f"{prefix}: {text}")

        return "\n".join(formatted_lines)

    def clear(self, session_id: str = "default") -> None:
        """Reset conversation history for a session."""
        self._sessions[session_id] = []


# Global Singleton Memory Store
memory_store = ConversationMemory(max_turns=6)
