"""
agents/retrieval.py - Retrieval Agent

Executes multi-query hybrid retrieval (Vector + BM25 + RRF) for all
sub-queries produced by the Query Planner Agent, then fuses and deduplicates
the results into a unified ranked list of document chunks.

No LLM required - pure deterministic retrieval logic.
Wraps the existing search_vector_store() from rag.py.

Low-latency optimisation: all sub-queries are dispatched in parallel via
ThreadPoolExecutor so total retrieval time equals the slowest single query,
not the sum of all queries.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Any

from rag import search_vector_store
from graph_state import AgentState

logger = logging.getLogger("retrieval_agent")

# k per sub-query; more sub-queries -> more total candidates before dedup
CHUNKS_PER_SUBQUERY = 5
 
# Max parallel workers for retrieval (I/O-bound, so threads are fine)
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")

# Per-subquery retrieval timeout (seconds)
_SUBQUERY_TIMEOUT = 10


def _run_subquery(query: str, session_id: str) -> List[Dict[str, Any]]:
    """Execute a single sub-query retrieval. Runs inside a thread pool worker."""
    return search_vector_store(query, k=CHUNKS_PER_SUBQUERY, session_id=session_id)


def retrieval_node(state: AgentState) -> dict:
    """
    Runs each sub-query through the hybrid vector+BM25 retriever IN PARALLEL,
    then deduplicates and returns the top-k chunks by RRF score.

    Parallel dispatch means total latency ≈ max(individual query latency)
    instead of sum(individual query latencies).
    """
    sub_queries = state.get("sub_queries") or [state["raw_query"]]
    session_id = state.get("session_id", "")
    logger.info(f"[Retrieval] Dispatching {len(sub_queries)} sub-queries in parallel for session {session_id}")

    all_chunks: List[Dict[str, Any]] = []
    seen_content: set = set()

    # Submit all sub-queries to the thread pool simultaneously
    future_map = {
        _RETRIEVAL_EXECUTOR.submit(_run_subquery, q, session_id): i
        for i, q in enumerate(sub_queries)
    }

    for future in as_completed(future_map, timeout=_SUBQUERY_TIMEOUT + 5):
        idx = future_map[future]
        try:
            results = future.result(timeout=_SUBQUERY_TIMEOUT)
            logger.info(f"[Retrieval] Sub-query {idx + 1} returned {len(results)} chunks")
            for chunk in results:
                # Deduplicate by content fingerprint (first 200 chars)
                key = chunk["content"][:200]
                if key not in seen_content:
                    seen_content.add(key)
                    all_chunks.append(chunk)
        except FuturesTimeoutError:
            logger.warning(f"[Retrieval] Sub-query {idx + 1} timed out after {_SUBQUERY_TIMEOUT}s")
        except Exception as e:
            logger.warning(f"[Retrieval] Sub-query {idx + 1} failed: {e}")

    # Sort by RRF score descending and cap at top 10 unique chunks
    all_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    top_chunks = all_chunks[:10]

    logger.info(f"[Retrieval] Retrieved {len(top_chunks)} unique chunks total")

    return {
        "retrieved_chunks": top_chunks,
        "agent_trace": ["retrieval"],
    }
