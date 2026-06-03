"""
agents/memory_agent.py - Memory Agent

Reads the MySQL conversation history for the current session,
identifies what past context is relevant to the current query,
and produces a compact memory summary to inject into Synthesis.

Model: FAST_MODEL (llama-3.1-8b-instant)
  - Module-level singleton avoids per-call ChatGroq construction.
  - max_tokens=256: output is a single short paragraph.
"""
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy.orm import Session

from config import settings
from graph_state import AgentState

logger = logging.getLogger("memory_agent")

# ── Module-level LLM singleton ────────────────────────────────────────────────
_llm = ChatGroq(
    model=settings.FAST_MODEL,
    temperature=0.0,
    groq_api_key=settings.GROQ_API_KEY,
    max_tokens=settings.MAX_TOKENS_ROUTING,   # short paragraph output
)

MEMORY_AGENT_PROMPT = """You are a conversation memory specialist for a RAG assistant.

Given recent conversation history and the user's current query, extract only the relevant past context.

Your output should be a compact paragraph (max 150 words) that:
1. Identifies what topics/files/answers were discussed that are relevant to the current query.
2. Resolves any pronouns - state what "it", "they", "the document", "that topic" refers to.
3. Notes any corrections or follow-ups from the user.

If there is NO relevant prior context, output exactly: "No relevant prior context."

Do NOT answer the user's question. Only summarize the relevant past context.
"""

# Max number of recent messages to feed to the Memory Agent
MAX_HISTORY_MESSAGES = 12


def memory_agent_node(state: AgentState, db_session: Session) -> dict:
    """
    Reads recent MySQL history and extracts a relevant memory summary.
    """
    if not state.get("requires_memory", False):
        logger.info("[Memory] Skipped (requires_memory=False)")
        return {"memory_summary": "", "agent_trace": ["memory_agent(skipped)"]}

    session_id = state.get("session_id", "")
    raw_query = state.get("raw_query", "")
    logger.info(f"[Memory] Building memory summary for session {session_id}")

    try:
        from memory import ChatMessageModel
        messages = (
            db_session.query(ChatMessageModel)
            .filter_by(session_id=session_id)
            .order_by(ChatMessageModel.id.desc())
            .limit(MAX_HISTORY_MESSAGES)
            .all()
        )
        messages = list(reversed(messages))

        if not messages:
            return {"memory_summary": "", "agent_trace": ["memory_agent(no_history)"]}

        # Format history as a readable transcript
        history_text = "\n".join(
            f"[{m.role.upper()}]: {m.content[:300]}" for m in messages
        )

        response = _llm.invoke([
            SystemMessage(content=MEMORY_AGENT_PROMPT),
            HumanMessage(content=f"Current Query: {raw_query}\n\nConversation History:\n{history_text}")
        ])

        summary = response.content.strip()
        if "no relevant" in summary.lower():
            summary = ""

        logger.info(f"[Memory] Summary length: {len(summary)} chars")

    except Exception as e:
        logger.warning(f"[Memory] Failed to build summary: {e}")
        summary = ""

    return {
        "memory_summary": summary,
        "agent_trace": ["memory_agent"],
    }
