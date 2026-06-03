"""
agents/query_planner.py - Query Planner Agent

Rewrites the raw user query into optimized retrieval sub-queries,
decomposes complex multi-part questions, and resolves pronouns using history.

Model: FAST_MODEL (llama-3.1-8b-instant)
  - Module-level singleton avoids per-call ChatGroq construction.
  - max_tokens=256: only a short JSON plan is expected.
"""
import json
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from graph_state import AgentState

logger = logging.getLogger("query_planner_agent")

# ── Module-level LLM singleton ────────────────────────────────────────────────
_llm = ChatGroq(
    model=settings.FAST_MODEL,
    temperature=0.1,
    groq_api_key=settings.GROQ_API_KEY,
    max_tokens=settings.MAX_TOKENS_ROUTING,   # JSON plan only
)

QUERY_PLANNER_PROMPT = """You are a Query Planning specialist for a RAG (Retrieval-Augmented Generation) system.

Your job is to analyze the user's query and produce optimized retrieval inputs.

Tasks:
1. REWRITE the query to be more specific and retrieval-friendly (expand acronyms, add context).
2. DECOMPOSE if the query has multiple distinct parts - split into separate sub-queries.
3. RESOLVE pronouns using the memory summary (if provided). Replace "it", "they", "this", "that" with their actual referents.
4. Decide if WEB SEARCH is required (only if user explicitly wants current/online info AND documents may be insufficient).

Output ONLY valid JSON (no markdown, no explanation):
{
  "standalone_query": "<pronoun-resolved version of the original query>",
  "sub_queries": ["<optimized query 1>", "<optimized query 2>"],
  "requires_web": <true|false>,
  "reasoning": "<brief explanation>"
}

Rules:
- Always produce at least 1 sub_query.
- Maximum 3 sub_queries (avoid over-decomposition).
- sub_queries should be diverse enough to retrieve complementary chunks.
- Never answer the question - only plan the retrieval.
"""


def query_planner_node(state: AgentState) -> dict:
    """
    Plans retrieval sub-queries from the raw user query.
    """
    raw_query = state["raw_query"]
    memory_summary = state.get("memory_summary", "")
    logger.info(f"[QueryPlanner] Planning for: {raw_query[:80]}...")

    memory_context = f"\nMemory (recent conversation context):\n{memory_summary}" if memory_summary else ""

    try:
        messages = [
            SystemMessage(content=QUERY_PLANNER_PROMPT),
            HumanMessage(content=f"User Query: {raw_query}{memory_context}")
        ]

        response = _llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        plan = json.loads(raw)
        sub_queries = plan.get("sub_queries", [raw_query])
        standalone = plan.get("standalone_query", raw_query)
        requires_web = bool(plan.get("requires_web", False))

        if not sub_queries:
            sub_queries = [raw_query]

        logger.info(f"[QueryPlanner] {len(sub_queries)} sub-queries, requires_web={requires_web}")

    except Exception as e:
        logger.warning(f"[QueryPlanner] Planning failed ({e}), using raw query")
        sub_queries = [raw_query]
        standalone = raw_query
        requires_web = False

    return {
        "sub_queries": sub_queries,
        "standalone_query": standalone,
        "requires_web": requires_web,
        "agent_trace": ["query_planner"],
    }
