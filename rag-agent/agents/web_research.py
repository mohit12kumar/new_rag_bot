"""
agents/web_research.py - Web Research Agent

Plans and executes targeted web searches when retrieval chunks are
insufficient. Anchors queries to the PDF topic context to prevent drift.
Only activated when requires_web=True or retrieved_chunks are sparse.

Model: FAST_MODEL (llama-3.1-8b-instant)
  - Module-level singleton avoids per-call ChatGroq construction.
  - max_tokens=64: output is just a single search query string.
"""
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from graph_state import AgentState

logger = logging.getLogger("web_research_agent")

# ── Module-level LLM singleton ────────────────────────────────────────────────
_llm = ChatGroq(
    model=settings.FAST_MODEL,
    temperature=0.0,
    groq_api_key=settings.GROQ_API_KEY,
    max_tokens=256,   # sufficient for validation and query planning
)

RELEVANCE_CHECK_PROMPT = """You are a relevance validator.
Given a user query and a few document snippets, determine if the query is related to the specific context of these document snippets.

Output EXACTLY:
'YES' if the query is related to the document snippets (even if asking for a detail not explicitly present, but still on the same topic/entity/concept).
'NO' if the query is unrelated, out-of-context, general knowledge (e.g. general coding, general history, chitchat, unrelated public entities).

Response (YES/NO):"""

WEB_RESEARCH_PROMPT_ANCHORED = """You are a web research planner for a document Q&A system.
Given a user query and relevant document snippets, output ONLY a single optimized web search query string that is anchored to the topics in the document snippets.
Do not include any explanations, quotes, or markdown.
"""

WEB_RESEARCH_PROMPT_GLOBAL = """You are a web search planner.
Given a user query, output ONLY a single optimized web search query string to look up the answer.
Do not include any explanations, quotes, or markdown.
"""


def web_research_node(state: AgentState) -> dict:
    """
    Runs a web search to supplement retrieval or search globally if out-of-context.
    """
    chunks = state.get("retrieved_chunks", [])
    requires_web = state.get("requires_web", False)
    query = state.get("standalone_query") or state.get("raw_query", "")

    # ── Check relevance to classify if query is out of context (global) ──
    is_global = False
    if not chunks:
        is_global = True
    else:
        try:
            snippet_text = "\n\n".join(f"Snippet:\n{c.get('content', '')[:300]}" for c in chunks[:3])
            response = _llm.invoke([
                SystemMessage(content=RELEVANCE_CHECK_PROMPT),
                HumanMessage(content=f"Document Snippets:\n{snippet_text}\n\nUser Query: {query}")
            ])
            decision = response.content.strip().upper()
            if "NO" in decision and "YES" not in decision:
                is_global = True
        except Exception as e:
            logger.warning(f"[WebResearch] Relevance check failed: {e}")
            is_global = False

    # Activation condition: run if global (out of context), explicitly requested, OR sparse document context
    if not is_global and not requires_web and len(chunks) >= 2:
        logger.info("[WebResearch] Skipped (sufficient document chunks, web not required, query in-context)")
        return {"web_snippets": [], "is_global": False, "agent_trace": ["web_research(skipped)"]}

    logger.info(f"[WebResearch] Running web research (is_global={is_global}) for: {query[:60]}...")

    # Plan search query via LLM singleton based on context-relevance
    anchored_query = query
    if is_global:
        try:
            response = _llm.invoke([
                SystemMessage(content=WEB_RESEARCH_PROMPT_GLOBAL),
                HumanMessage(content=f"User query: {query}")
            ])
            planned_query = response.content.strip().strip('"')
            if planned_query:
                anchored_query = planned_query
        except Exception as e:
            logger.warning(f"[WebResearch] Global query planning failed: {e}")
    else:
        try:
            snippet_text = "\n\n".join(f"- {c.get('content', '')[:300]}" for c in chunks[:3])
            response = _llm.invoke([
                SystemMessage(content=WEB_RESEARCH_PROMPT_ANCHORED),
                HumanMessage(content=f"User query: {query}\n\nDocument snippets:\n{snippet_text}")
            ])
            planned_query = response.content.strip().strip('"')
            if planned_query:
                anchored_query = planned_query
        except Exception as e:
            logger.warning(f"[WebResearch] Anchored query planning failed: {e}")

    # Execute DuckDuckGo search
    web_snippets = []
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        search = DuckDuckGoSearchAPIWrapper()
        results = search.results(anchored_query, max_results=4)

        for r in results:
            snippet = r.get("snippet", "")
            if snippet:
                web_snippets.append({
                    "source": r.get("title", "Web Result"),
                    "url": r.get("link", ""),
                    "snippet": snippet[:400]
                })

        logger.info(f"[WebResearch] Got {len(web_snippets)} web snippets")
    except Exception as e:
        logger.warning(f"[WebResearch] Search failed: {e}")

    return {
        "web_snippets": web_snippets,
        "is_global": is_global,
        "agent_trace": ["web_research"],
    }
