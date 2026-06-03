"""
agents/supervisor.py - Supervisor Agent (Orchestrator)

The Supervisor is the entry point of the LangGraph. It classifies the user's
intent and routes to the appropriate specialist agents in the correct order.

Model: SUPERVISOR_MODEL (llama-3.1-8b-instant)
  - Routing only; no 70B model needed.
  - Module-level singleton avoids re-instantiation on every request.
  - max_tokens=256: only short JSON is expected, tighter cap = faster decode.
"""
import json
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from graph_state import AgentState

logger = logging.getLogger("supervisor_agent")

# ── Module-level LLM singleton ────────────────────────────────────────────────
# Instantiated once at import time; reused across all requests.
_llm = ChatGroq(
    model=settings.SUPERVISOR_MODEL,
    temperature=0.0,
    groq_api_key=settings.GROQ_API_KEY,
    max_tokens=settings.MAX_TOKENS_ROUTING,      # short JSON output only
)

SUPERVISOR_SYSTEM_PROMPT = """You are a multi-agent RAG orchestrator. Your ONLY job is to classify the user's intent and output a JSON routing decision.

Intents:
- "chitchat"      : Greetings, thanks, small talk, or anything unrelated to documents.
- "time"          : User asks for current date/time.
- "doc_qa"        : User asks a question that should be answered from uploaded documents.
- "web_research"  : User explicitly asks for web/online/latest information AND documents are insufficient.

Output ONLY valid JSON (no markdown, no explanation):
{
  "intent": "<intent>",
  "requires_memory": <true|false>,
  "reasoning": "<one sentence why>"
}

Rules:
- Default to "doc_qa" for any substantive question.
- "requires_memory" = true only when the query contains pronouns (it, they, that, this) or references previous turns.
- Never output anything except the JSON object.
"""


def supervisor_node(state: AgentState) -> dict:
    """
    Classifies the user query intent and sets routing flags.
    Returns partial state updates.
    """
    logger.info(f"[Supervisor] Classifying query: {state['raw_query'][:80]}...")

    try:
        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=f"Current time: {state.get('current_time', 'unknown')}\nUser query: {state['raw_query']}")
        ]

        response = _llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        decision = json.loads(raw)
        intent = decision.get("intent", "doc_qa")
        requires_memory = bool(decision.get("requires_memory", False))
        logger.info(f"[Supervisor] Intent={intent}, requires_memory={requires_memory}")

    except Exception as e:
        logger.warning(f"[Supervisor] Classification failed ({e}), defaulting to doc_qa")
        intent = "doc_qa"
        requires_memory = False

    return {
        "intent": intent,
        "requires_memory": requires_memory,
        "requires_web": False,       # Web Research Agent sets this if needed
        "agent_trace": ["supervisor"],
    }


def supervisor_route(state: AgentState) -> str:
    """
    LangGraph conditional edge: decides which node to run next after supervisor.
    """
    intent = state.get("intent", "doc_qa")
    if intent == "chitchat":
        return "synthesis"      # Skip retrieval; Synthesis will handle greetings
    elif intent == "time":
        return "synthesis"      # Synthesis reads current_time from state
    else:
        return "query_planner"  # doc_qa / web_research -> plan & retrieve
