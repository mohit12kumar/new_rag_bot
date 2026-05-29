import datetime
import contextvars
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from rag import search_vector_store

# Input schemas for tools to ensure strict parameter typing and description generation for Groq models
class SearchDocumentsInput(BaseModel):
    query: str = Field(description="The query text to search for in the document database.")

class WebSearchInput(BaseModel):
    query: str = Field(
        description=(
            "A search query to look up supplemental information on the web. "
            "ONLY use this when the user's question is about an uploaded PDF topic "
            "and the document search returned insufficient results. "
            "Do NOT use for general knowledge or unrelated questions."
        );.
    )

class GetCurrentTimeInput(BaseModel):
    placeholder: str = Field(default="", description="An optional placeholder argument, leave empty.")

# Thread-safe context variable to capture citations retrieved during a single API request run
retrieved_citations: contextvars.ContextVar[List[Dict[str, Any]]] = contextvars.ContextVar("retrieved_citations", default=[])
# Thread-safe context variable to capture current active session ID during an execution
current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_session_id", default="")

@tool("search_documents", args_schema=SearchDocumentsInput)
def search_documents(query: str) -> str:
    """
    Search the uploaded document database (knowledge base) for context matching the query.
    Use this tool to look up details in manuals, documents, code files, or uploaded text resources.
    """
    try:
        results = search_vector_store(query, k=5, session_id=current_session_id.get())
        if not results:
            return "No matching information found in the document database."
        
        # Capture citations in context variable for app.py to extract and return to the UI
        citations = []
        seen = set()
        for res in results:
            # Create a unique key for deduplication
            key = (res["source"], res["page"])
            if key not in seen:
                seen.add(key)
                citations.append({
                    "source": res["source"],
                    "page": res["page"],
                    "snippet": res["content"][:200] + "..."  # Truncated preview
                })
        
        # Merge or append to existing citations in this request context
        existing = retrieved_citations.get()
        retrieved_citations.set(existing + citations)
        
        # Format the response to be fed into the LLM context
        formatted = []
        for i, res in enumerate(results, 1):
            formatted.append(
                f"Document Source: {res['source']} (Page {res['page']})\n"
                f"Content Snippet: {res['content']}\n"
                f"---"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Error executing document database search: {str(e)}"

@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """
    FALLBACK ONLY: Search the web for supplemental details about a topic that EXISTS in
    the uploaded PDF documents but where the document search returned incomplete context.

    Rules (STRICTLY enforced):
    - MUST only be called after `search_documents` already returned at least one result.
    - MUST NOT be called for general knowledge, news, coding, or any off-PDF topic.
    - The search query is automatically anchored to the PDF's content — it cannot drift
      to unrelated topics.
    """
    # ── Gate: block if no PDF results exist for this request ────────────────────
    citations = retrieved_citations.get()
    if not citations:
        return (
            "⚠️ Web search is not allowed for this question. "
            "This assistant only answers questions about your uploaded PDF documents. "
            "No matching content was found in the uploaded files, so this topic is "
            "outside the scope of your documents. Please upload a relevant PDF or "
            "ask something related to your existing uploads."
        )

    # ── Gate: verify if the search query is related to the PDF content using LLM ──
    snippet_text = " ".join(c.get("snippet", "") for c in citations)
    try:
        from config import settings
        from langchain_groq import ChatGroq

        # Initialize LLM specifically for strict relevance verification
        relevance_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            groq_api_key=settings.GROQ_API_KEY
        )

        verification_prompt = (
            "You are a strict relevance validator. Your task is to determine if a proposed web search query is directly related to the content of the uploaded PDF snippets.\n\n"
            "Uploaded PDF Snippets:\n"
            "-------------------\n"
            f"{snippet_text}\n"
            "-------------------\n\n"
            "Proposed Web Search Query:\n"
            f"{query}\n\n"
            "Instructions:\n"
            "1. Analyze if the Proposed Web Search Query is related to the specific topics, concepts, events, or entities described in the Uploaded PDF Snippets.\n"
            "2. If the query is related to the PDF content (even if asking for a supplemental detail, clarification, or elaboration), output EXACTLY: 'YES'\n"
            "3. If the query is unrelated (e.g. general knowledge chitchat, coding help, other public figures/events not in the snippets, or unrelated topics), output EXACTLY: 'NO'\n\n"
            "Relevance (YES/NO):"
        )

        response = relevance_llm.invoke(verification_prompt)
        decision = response.content.strip().upper()

        if "NO" in decision and "YES" not in decision:
            return (
                "⚠️ Web search is not allowed because the query is not related to the uploaded PDF content. "
                "This assistant only answers questions about topics present in your uploaded PDF documents."
            )
    except Exception as e:
        # Fallback to keyword matching if LLM verification encounters an error
        pass

    # ── Build a PDF-context-anchored query ──────────────────────────────────────
    # Extract the top unique keywords from the retrieved PDF citation snippets.
    # These keywords ensure the web search stays focused on the PDF's actual subject.
    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "of", "in", "on", "at",
        "to", "for", "with", "by", "from", "and", "or", "but", "if", "that",
        "this", "it", "its", "as", "not", "no", "so", "than", "then", "when",
        "where", "which", "who", "whom", "what", "how", "all", "each", "both",
        "such", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "up", "about", "against", "also", "just",
        "any", "more", "other", "same", "only", "over", "under", "again",
    }

    # Collect text from the first 3 citation snippets
    snippet_text_for_terms = " ".join(c.get("snippet", "")[:200] for c in citations[:3])
    # Extract meaningful words (length > 3, not stopwords, alphanumeric only)
    raw_words = [
        w.strip(".,;:!?\"'()[]") for w in snippet_text_for_terms.split()
    ]
    key_terms = []
    seen_terms: set = set()
    for w in raw_words:
        lower = w.lower()
        if len(lower) > 3 and lower not in _STOPWORDS and lower.isalpha() and lower not in seen_terms:
            seen_terms.add(lower)
            key_terms.append(w)
        if len(key_terms) >= 8:
            break

    # Combine original query with PDF-derived key terms
    pdf_context = " ".join(key_terms)
    anchored_query = f"{query} {pdf_context}".strip()

    # ── Execute the anchored web search ─────────────────────────────────────────
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        result = search.run(anchored_query)
        return f"[Web search anchored to PDF topics: {pdf_context}]\n\n{result}"
    except Exception as e:
        return f"Web search is currently unavailable: {str(e)}"

@tool("get_current_time", args_schema=GetCurrentTimeInput)
def get_current_time(placeholder: str = "") -> str:
    """
    Get the current date and time. Use this when the user asks about the current day,
    current time, or relative dates (e.g. 'what happened last week?').
    """
    return datetime.datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
