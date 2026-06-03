"""
agents/synthesis.py - Synthesis Agent

Takes all gathered context (document chunks, web snippets, memory summary)
and produces the final grounded, cited Markdown answer.

Model: SYNTHESIS_MODEL (llama-3.1-8b-instant)
  - Switched from 70B to 8B for low latency; quality preserved via prompt.
  - Module-level singleton avoids per-call ChatGroq construction.
  - max_tokens=1500: enough headroom for a complete cited answer.
"""
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict, Any

from config import settings
from graph_state import AgentState

logger = logging.getLogger("synthesis_agent")

# ── Module-level LLM singleton ────────────────────────────────────────────────
_llm = ChatGroq(
    model=settings.SYNTHESIS_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    groq_api_key=settings.GROQ_API_KEY,
    max_tokens=settings.MAX_TOKENS_SYNTHESIS,  # headroom for full cited answer
)

SYNTHESIS_SYSTEM_PROMPT_IN_CONTEXT = """You are a precise Document Q&A Assistant. Your goal is to produce a complete, accurate, and well-cited answer.

Guidelines:
1. Answer using the provided context (document chunks + web snippets + memory).
2. Cite every fact from the documents: use [Source: filename.pdf, Page X]. Do NOT cite web snippets or use web URLs in citations.
3. Format using Markdown: use headers, bullet points, and bold for clarity.
4. Do NOT hallucinate - only state what is explicitly supported by the context.
"""

SYNTHESIS_SYSTEM_PROMPT_GLOBAL = """You are a helpful and knowledgeable AI Assistant.

Guidelines:
1. Since the user's query is out of context of the uploaded documents, you must answer the query globally based on the provided web snippets or your own knowledge.
2. DO NOT use any citations in your answer (no [Source: ...], no [Web: ...], etc.). The answer should be clean, natural, and free of citations.
3. Format using Markdown: use headers, bullet points, and bold for clarity.
4. Do NOT output any warning messages about the topic not being in the uploaded documents. Just answer the query directly.
"""


def _build_context_block(chunks: List[Dict], web_snippets: List[Dict], memory: str) -> str:
    """Formats all context sources into a structured block for the LLM."""
    parts = []

    if memory:
        parts.append(f"## Conversation Memory\n{memory}")

    if chunks:
        doc_parts = []
        for i, c in enumerate(chunks, 1):
            doc_parts.append(
                f"[{i}] Source: {c.get('source','Unknown')} | Page: {c.get('page',1)}\n"
                f"{c.get('content','')}"
            )
        parts.append("## Document Context\n" + "\n\n".join(doc_parts))

    if web_snippets:
        web_parts = []
        for w in web_snippets:
            web_parts.append(
                f"Web: {w.get('source','Web')} ({w.get('url','')})\n{w.get('snippet','')}"
            )
        parts.append("## Web Research Context\n" + "\n\n".join(web_parts))

    return "\n\n---\n\n".join(parts) if parts else ""


def synthesis_node(state: AgentState) -> dict:
    """
    Generates the final answer from all gathered context.
    """
    intent = state.get("intent", "doc_qa")
    query = state.get("standalone_query") or state.get("raw_query", "")
    chunks = state.get("retrieved_chunks", [])
    web_snippets = state.get("web_snippets", [])
    memory = state.get("memory_summary", "")
    current_time = state.get("current_time", "")
    is_global = state.get("is_global", False)

    logger.info(f"[Synthesis] Generating answer. chunks={len(chunks)}, web={len(web_snippets)}, intent={intent}, is_global={is_global}")

    # Build citations list for the API response (PDF ONLY, and only if not global)
    citations: List[Dict[str, Any]] = []
    if not is_global:
        seen_cit: set = set()
        for c in chunks:
            key = (c.get("source",""), c.get("page", 1))
            if key not in seen_cit:
                seen_cit.add(key)
                citations.append({
                    "source": c.get("source","Unknown"),
                    "page": c.get("page", 1),
                    "snippet": c.get("content","")[:200] + "..."
                })

    try:
        context_block = _build_context_block(chunks, web_snippets, memory)

        if intent in ("chitchat",):
            system_prompt = "You are a helpful and warm AI assistant. Answer the user's message directly and naturally."
            user_msg = f"The user says: {query}"
        elif intent == "time":
            system_prompt = "You are an AI assistant. Tell the user the current date and time directly."
            user_msg = f"The user asks: {query}\n\nCurrent time: {current_time}"
        elif is_global:
            system_prompt = SYNTHESIS_SYSTEM_PROMPT_GLOBAL
            user_msg = (
                f"User Question: {query}\n\n"
                f"Context:\n{context_block if context_block else 'No context retrieved.'}"
            )
        else:
            system_prompt = SYNTHESIS_SYSTEM_PROMPT_IN_CONTEXT
            user_msg = (
                f"User Question: {query}\n\n"
                f"Context:\n{context_block if context_block else 'No context retrieved.'}"
            )

        response = _llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg)
        ])

        answer = response.content.strip()
        logger.info(f"[Synthesis] Answer length: {len(answer)} chars")

    except Exception as e:
        logger.error(f"[Synthesis] Failed: {e}", exc_info=True)
        answer = f"[ERROR] An error occurred while generating the answer: {str(e)}"

    return {
        "draft_answer": answer,
        "citations": citations,
        "agent_trace": ["synthesis"],
    }
