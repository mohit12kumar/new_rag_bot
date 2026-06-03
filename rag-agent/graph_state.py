"""
graph_state.py - Shared state schema for the Multi-Agent RAG LangGraph.

All agents read from and write into this single AgentState TypedDict,
which flows through the LangGraph StateGraph as the message bus.
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class AgentState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    session_id: str
    raw_query: str
    current_time: str
    model: Optional[str]          # Optional user-selected model override

    # ── Query Planner output ─────────────────────────────────────────────────
    intent: str                   # "doc_qa" | "chitchat" | "time" | "web_research"
    sub_queries: List[str]        # Rewritten / decomposed retrieval queries
    requires_web: bool
    requires_memory: bool
    standalone_query: str         # Pronoun-resolved version of raw_query

    # ── Retrieval Agent output ───────────────────────────────────────────────
    retrieved_chunks: List[Dict[str, Any]]   # [{content, source, page, score}]

    # ── Web Research Agent output ────────────────────────────────────────────
    web_snippets: List[Dict[str, Any]]       # [{title, url, snippet}]

    # ── Memory Agent output ──────────────────────────────────────────────────
    memory_summary: str           # Compact relevant past-turn summary

    # ── Synthesis Agent output ───────────────────────────────────────────────
    draft_answer: str             # Generated answer (pre-critique)

    # ── Critique Agent output ────────────────────────────────────────────────
    critique_score: float         # 0.0 – 1.0 confidence
    critique_notes: str           # Explanation of gaps / issues
    approved: bool                # True when score >= threshold

    # ── Final output ─────────────────────────────────────────────────────────
    final_answer: str
    citations: List[Dict[str, Any]]
    agent_trace: Annotated[List[str], operator.add]  # Accumulates agent names
    retry_count: int              # How many re-retrieval loops happened
    is_global: bool               # True if query is out-of-context and answered globally

