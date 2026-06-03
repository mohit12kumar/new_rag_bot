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
        )
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
        retrieved_citations.get().extend(citations)
        
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
    Search the web for supplemental details.
    - If the query is related to the PDF context, the query is optimized based on PDF snippets.
    - If the query is out-of-context, the search is executed globally.
    """
    from config import settings
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    # ── Initialize LLM ──
    try:
        relevance_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            groq_api_key=settings.GROQ_API_KEY,
            max_tokens=256
        )
    except Exception as e:
        return f"Failed to initialize LLM for search planning: {str(e)}"

    # Check if we have PDF citations in context
    pdf_citations = retrieved_citations.get()
    
    is_global = True
    snippet_text = ""
    if pdf_citations:
        snippet_text = " ".join(c.get("snippet", "") for c in pdf_citations)
        # Check relevance to determine if we should anchor or search globally
        try:
            verification_prompt = (
                "You are a relevance validator. Your task is to determine if a user query is related to the content of the uploaded PDF snippets.\n\n"
                "Uploaded PDF Snippets:\n"
                f"{snippet_text}\n\n"
                "User Query:\n"
                f"{query}\n\n"
                "Instructions:\n"
                "1. Output EXACTLY 'YES' if the query is related to the PDF content.\n"
                "2. Output EXACTLY 'NO' if the query is unrelated / out of context.\n"
                "Relevance (YES/NO):"
            )
            response = relevance_llm.invoke(verification_prompt)
            decision = response.content.strip().upper()
            if "YES" in decision:
                is_global = False
        except Exception:
            is_global = False

    # ── Build the search query ──
    anchored_query = query
    if is_global:
        try:
            response = relevance_llm.invoke([
                SystemMessage(content="You are a web search planner. Given a user query, output ONLY the single optimized web search query string to look up the answer. Do not include any explanations, quotes, or markdown."),
                HumanMessage(content=f"User query: {query}")
            ])
            planned_query = response.content.strip().strip('"')
            if planned_query:
                anchored_query = planned_query
        except Exception:
            pass
    else:
        try:
            response = relevance_llm.invoke([
                SystemMessage(content="You are a web search planner for a document Q&A system. Given a user query and relevant document snippets, output ONLY a single optimized web search query string that is anchored to the topics in the document snippets. Do not include explanations, quotes, or markdown."),
                HumanMessage(content=f"User query: {query}\n\nDocument snippets:\n{snippet_text}")
            ])
            planned_query = response.content.strip().strip('"')
            if planned_query:
                anchored_query = planned_query
        except Exception:
            pass

    # ── Execute the search ──
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        search = DuckDuckGoSearchAPIWrapper()
        results = search.results(anchored_query, max_results=5)
        
        if not results:
            return f"[Web search for query: {anchored_query}]\n\nNo search results found."
            
        formatted = []
        for r in results:
            snippet = r.get("snippet", "")
            title = r.get("title", "")
            link = r.get("link", "")
            
            if not snippet:
                continue
                
            formatted.append(
                f"Web Title: {title}\n"
                f"Web URL: {link}\n"
                f"Snippet: {snippet}\n"
                f"---"
            )
            
        # We do NOT append web snippets to the retrieved_citations context variables
        # so that only PDF citations are returned and rendered.
        
        return f"[Web search results for: {anchored_query}]\n\n" + "\n\n".join(formatted)
    except Exception as e:
        return f"Web search is currently unavailable: {str(e)}"

@tool("get_current_time", args_schema=GetCurrentTimeInput)
def get_current_time(placeholder: str = "") -> str:
    """
    Get the current date and time. Use this when the user asks about the current day,
    current time, or relative dates (e.g. 'what happened last week?').
    """
    return datetime.datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
