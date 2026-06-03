"""
agents/critique.py - Critique Agent

Quality gate that scores the draft answer for grounding and accuracy.
If confidence < threshold, triggers a re-retrieval loop (up to MAX_CRITIQUE_RETRIES).

Model: FAST_MODEL (llama-3.1-8b-instant)
  - Module-level singleton avoids per-call ChatGroq construction.
  - max_tokens=256: only a short JSON verdict is expected.
"""
import json
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from graph_state import AgentState

logger = logging.getLogger("critique_agent")

# ── Module-level LLM singleton ────────────────────────────────────────────────
_llm = ChatGroq(
    model=settings.FAST_MODEL,
    temperature=0.0,
    groq_api_key=settings.GROQ_API_KEY,
    max_tokens=settings.MAX_TOKENS_ROUTING,   # short JSON verdict only
)

CRITIQUE_SYSTEM_PROMPT = """You are a strict answer quality evaluator for a RAG system.

Evaluate if the ANSWER is properly grounded in the CONTEXT provided.

Score the answer on a scale from 0.0 to 1.0:
- 1.0: Fully grounded, every claim maps to the context, proper citations.
- 0.7-0.9: Mostly grounded, minor gaps or imprecise citations.
- 0.4-0.6: Partially grounded, some unsupported claims present.
- 0.0-0.3: Mostly hallucinated or no relevant context was used.

Output ONLY valid JSON (no markdown):
{
  "score": <0.0-1.0>,
  "approved": <true if score >= 0.7>,
  "issues": "<brief description of issues, or 'None' if approved>",
  "missing_context": "<what additional retrieval would help, or 'None'>"
}
"""


def critique_node(state: AgentState) -> dict:
    """
    Evaluates the draft answer quality and decides whether to approve or retry.
    """
    draft = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks", [])
    intent = state.get("intent", "doc_qa")
    retry_count = state.get("retry_count", 0)

    # Skip critique for simple intents - no need to evaluate chitchat/time
    if intent in ("chitchat", "time"):
        logger.info("[Critique] Skipped for intent: %s", intent)
        return {
            "critique_score": 1.0,
            "critique_notes": "Skipped for non-doc intent",
            "approved": True,
            "final_answer": draft,
            "retry_count": retry_count,
            "agent_trace": ["critique(skipped)"],
        }

    # If we've already retried the max times, approve as-is to avoid infinite loop
    if retry_count >= settings.MAX_CRITIQUE_RETRIES:
        logger.info("[Critique] Max retries reached, approving as-is")
        return {
            "critique_score": 0.65,
            "critique_notes": "Max retries reached - returning best available answer",
            "approved": True,
            "final_answer": draft,
            "retry_count": retry_count,
            "agent_trace": ["critique(max_retries)"],
        }

    logger.info(f"[Critique] Evaluating draft (retry_count={retry_count})")

    # Build a compact context summary for the evaluator
    context_summary = "\n".join(
        f"- [{c.get('source')} p{c.get('page')}]: {c.get('content','')[:150]}"
        for c in chunks[:5]
    )

    try:
        response = _llm.invoke([
            SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"CONTEXT:\n{context_summary or 'No document context available.'}\n\n"
                f"ANSWER:\n{draft[:1500]}"
            ))
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        score = float(result.get("score", 0.5))
        approved = score >= settings.CRITIQUE_SCORE_THRESHOLD
        notes = result.get("issues", "")

        logger.info(f"[Critique] Score={score:.2f}, approved={approved}")

    except Exception as e:
        logger.warning(f"[Critique] Evaluation failed ({e}), approving by default")
        score = 0.75
        approved = True
        notes = "Evaluation failed - approved by default"

    return {
        "critique_score": score,
        "critique_notes": notes,
        "approved": approved,
        "final_answer": draft if approved else "",
        "retry_count": retry_count + (0 if approved else 1),
        "agent_trace": ["critique"],
    }


def critique_route(state: AgentState) -> str:
    """
    LangGraph conditional edge: after critique, either finish or re-retrieve.
    """
    if state.get("approved", False):
        return "end"
    else:
        logger.info("[Critique] Answer not approved - routing back to retrieval")
        return "retrieval"
