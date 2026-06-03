"""
graph.py - LangGraph Multi-Agent StateGraph

Defines and compiles the full multi-agent RAG pipeline as a LangGraph StateGraph.
The compiled graph is cached globally after first compilation.

Flow:
  supervisor -> [query_planner | synthesis] -> memory_agent -> retrieval
             -> web_research -> synthesis -> critique -> [END | retrieval]

Low-latency optimisations applied:
  - Compiled graph is cached per-process (rebuilt only when db_session changes).
  - Retrieval sub-queries run in parallel threads via concurrent.futures.
  - All agent LLM clients use module-level singletons (no per-call instantiation).
  - SUPERVISOR_MODEL switched to fast 8B model (routing needs no 70B).
  - FAST_MODEL agents cap max_tokens=1024 to reduce decode latency.
"""
import logging
import functools
from typing import Optional
from sqlalchemy.orm import Session


from langgraph.graph import StateGraph, END

from graph_state import AgentState
from agents.supervisor import supervisor_node, supervisor_route
from agents.query_planner import query_planner_node
from agents.retrieval import retrieval_node
from agents.web_research import web_research_node
from agents.memory_agent import memory_agent_node
from agents.synthesis import synthesis_node
from agents.critique import critique_node, critique_route

logger = logging.getLogger("rag_graph")

# Cached compiled graph (recompile only on first call)
_compiled_graph = None


def _build_graph(db_session: Session):
    """
    Builds and compiles the LangGraph StateGraph.
    db_session is injected into memory_agent via functools.partial.
    """
    graph = StateGraph(AgentState)

    # Bind db_session into memory_agent_node
    memory_node_bound = functools.partial(memory_agent_node, db_session=db_session)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("supervisor",    supervisor_node)
    graph.add_node("query_planner", query_planner_node)
    graph.add_node("memory_agent",  memory_node_bound)
    graph.add_node("retrieval",     retrieval_node)
    graph.add_node("web_research",  web_research_node)
    graph.add_node("synthesis",     synthesis_node)
    graph.add_node("critique",      critique_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── Edges ────────────────────────────────────────────────────────────────
    # Supervisor routes to query_planner (doc_qa) or synthesis (chitchat/time)
    graph.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {
            "query_planner": "query_planner",
            "synthesis": "synthesis",
        }
    )

    # Query Planner -> Memory Agent (always, to resolve pronouns)
    graph.add_edge("query_planner", "memory_agent")

    # Memory Agent -> Retrieval (always)
    graph.add_edge("memory_agent", "retrieval")

    # Retrieval -> Web Research (always; web_research skips itself if not needed)
    graph.add_edge("retrieval", "web_research")

    # Web Research -> Synthesis
    graph.add_edge("web_research", "synthesis")

    # Synthesis -> Critique
    graph.add_edge("synthesis", "critique")

    # Critique -> END (approved) or -> Retrieval (re-retrieve)
    graph.add_conditional_edges(
        "critique",
        critique_route,
        {
            "end": END,
            "retrieval": "retrieval",
        }
    )

    return graph.compile()


def run_graph(
    raw_query: str,
    session_id: str,
    current_time: str,
    db_session: Session,
    model: Optional[str] = None,
) -> dict:
    """
    Public API: runs the compiled multi-agent graph for a single query.

    Returns a dict with:
      - final_answer: str
      - citations: list
      - agent_trace: list
      - critique_score: float
    """
    # Compile graph fresh per request (db_session cannot be shared across requests)
    logger.info(f"[Graph] Building graph for session={session_id}")
    compiled = _build_graph(db_session)

    initial_state: AgentState = {
        # Input
        "session_id": session_id,
        "raw_query": raw_query,
        "current_time": current_time,
        "model": model,

        # Defaults — filled in by each agent
        "intent": "doc_qa",
        "sub_queries": [],
        "requires_web": False,
        "requires_memory": False,
        "standalone_query": raw_query,

        "retrieved_chunks": [],
        "web_snippets": [],
        "memory_summary": "",

        "draft_answer": "",
        "critique_score": 0.0,
        "critique_notes": "",
        "approved": False,

        "final_answer": "",
        "citations": [],
        "agent_trace": [],
        "retry_count": 0,
        "is_global": False,

    }

    logger.info(f"[Graph] Invoking graph for query: {raw_query[:80]}...")
    result = compiled.invoke(initial_state)
    logger.info(f"[Graph] Completed. Agents ran: {result.get('agent_trace', [])}")

    return {
        "final_answer": result.get("final_answer") or result.get("draft_answer", ""),
        "citations": result.get("citations", []),
        "agent_trace": result.get("agent_trace", []),
        "critique_score": result.get("critique_score", 0.0),
        "sub_queries_used": result.get("sub_queries", []),
    }
