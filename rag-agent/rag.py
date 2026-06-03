import os
from typing import List, Dict, Any
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from config import settings
from exceptions import InvalidDocumentError, VectorStoreError

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize and return HuggingFace local Embeddings.
    Does not require any API keys. Cached to prevent repeated model loading.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """
    Initialize and return the Chroma vector store instance.
    Cached as a singleton — Chroma client creation is expensive and
    the same instance can be safely reused across all requests.
    """
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=settings.CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name="rag_agent_collection"
    )

def ingest_file(file_path: str, session_id: str = None) -> int:
    """
    Load a file, split its text into chunks, remove any pre-existing chunks of the same file, 
    and store the new chunks in Chroma DB.
    
    Returns the number of document chunks generated.
    """
    if not os.path.exists(file_path):
        raise InvalidDocumentError(f"File not found at path: {file_path}")

    filename = os.path.basename(file_path)
    
    # 1. Load File content
    try:
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif filename.lower().endswith((".txt", ".md")):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        else:
            raise InvalidDocumentError(
                message=f"Unsupported file format for {filename}. Only PDF, TXT, and MD files are supported."
            )
    except InvalidDocumentError:
        raise
    except Exception as e:
        raise InvalidDocumentError(
            message=f"Failed to parse document '{filename}'. The file may be corrupt or unreadable.",
            details=str(e)
        )

    # 2. Normalize metadata (ensure source points to filename and page exists)
    try:
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata["session_id"] = session_id if session_id else "NO_SESSION_DEFINED"
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1
            else:
                # 1-index pages for readability
                doc.metadata["page"] = doc.metadata["page"] + 1 if isinstance(doc.metadata["page"], int) else 1
    except Exception as e:
        raise InvalidDocumentError(
            message=f"Failed to process metadata for document '{filename}'.",
            details=str(e)
        )

    # 3. Chunk text recursively
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = splitter.split_documents(docs)
    except Exception as e:
        raise InvalidDocumentError(
            message=f"Failed to split text of document '{filename}' into chunks.",
            details=str(e)
        )

    if not chunks:
        raise InvalidDocumentError(
            message=f"Document '{filename}' resulted in 0 text chunks. Ensure the file is not empty."
        )

    # 4. Remove previous indexes of this file to prevent duplicates
    try:
        delete_file_from_store(filename, session_id=session_id)
    except Exception as e:
        print(f"Warning: Failed to clean up old vectors for '{filename}': {e}")

    # 5. Insert documents into vector store
    try:
        store = get_vector_store()
        store.add_documents(chunks)
    except Exception as e:
        raise VectorStoreError(
            message=f"Failed to insert chunks of document '{filename}' into the vector store database.",
            details=str(e)
        )
    
    return len(chunks)

def delete_file_from_store(filename: str, session_id: str = None) -> None:
    """
    Deletes all chunks associated with a specific file from the vector database.
    """
    try:
        store = get_vector_store()
        # Strictly enforce deleting only from the specific session
        target_session = session_id if session_id else "NO_SESSION_DEFINED"
        where_filter = {
            "$and": [
                {"source": filename},
                {"session_id": target_session}
            ]
        }
        store._collection.delete(where=where_filter)
    except Exception as e:
        print(f"Warning: Failed to delete {filename} from Chroma: {e}")

def search_vector_store(query: str, k: int = 5, session_id: str = None) -> List[Dict[str, Any]]:
    """
    Perform a hybrid search (Vector + BM25) in Chroma and format the results using Reciprocal Rank Fusion (RRF).
    
    Returns a list of structured results containing:
      - content: Text snippet
      - source: Filename
      - page: Page number (if available)
      - score: Unified RRF score (higher is more relevant)
    """
    try:
        store = get_vector_store()
        # Strictly enforce search filtering by session_id to protect other sessions
        active_session = session_id if session_id else "NO_SESSION_DEFINED"
        search_filter = {"session_id": active_session}
        # Search with score (returns tuple of (Document, score))
        # Chroma returns L2 distance; lower is closer (more similar)
        raw_vector_results = store.similarity_search_with_score(query, k=k, filter=search_filter)
    except Exception as e:
        raise VectorStoreError(
            message="Failed to perform query search in the vector database.",
            details=str(e)
        )

    # Perform BM25 Search on session's documents
    raw_bm25_results = []
    try:
        collection_data = store._collection.get(
            where={"session_id": active_session},
            include=["documents", "metadatas"]
        )
        documents_list = collection_data.get("documents", []) if collection_data else []
        metadatas_list = collection_data.get("metadatas", []) if collection_data else []
        
        if documents_list:
            from langchain_core.documents import Document
            from langchain_community.retrievers import BM25Retriever
            
            docs = []
            for doc_text, meta in zip(documents_list, metadatas_list):
                docs.append(Document(page_content=doc_text, metadata=meta or {}))
                
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = k
            raw_bm25_results = bm25_retriever.invoke(query)
    except Exception as e:
        # Graceful fallback: log warning and proceed with vector results only
        print(f"Warning: Failed to perform BM25 search (falling back to vector-only): {e}")

    try:
        # Reciprocal Rank Fusion (RRF) algorithm to combine vector and BM25 rankings
        # Standard constant c = 60
        c = 60
        rrf_scores = {}
        doc_map = {}

        # 1. Fuse vector search results
        for rank, (doc, vec_score) in enumerate(raw_vector_results, 1):
            key = doc.page_content
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (c + rank))

        # 2. Fuse BM25 search results
        for rank, doc in enumerate(raw_bm25_results, 1):
            key = doc.page_content
            if key not in doc_map:
                doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (c + rank))

        # 3. Sort by RRF score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        formatted_results = []
        for key in sorted_keys[:k]:
            doc = doc_map[key]
            formatted_results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "score": float(rrf_scores[key])
            })
        return formatted_results
    except Exception as e:
        raise VectorStoreError(
            message="Failed to process search results from the hybrid retriever.",
            details=str(e)
        )


def get_indexed_files(session_id: str = None) -> List[str]:
    """
    Retrieve list of unique filenames that are currently indexed in Chroma.

    - If session_id is given: returns only files belonging to that session.
    - If session_id is None (e.g. called from /api/status): returns ALL
      indexed files across every session so the global counter is correct.
    """
    try:
        store = get_vector_store()

        if session_id:
            # Scoped query — only fetch metadata for this session
            collection_data = store._collection.get(
                where={"session_id": session_id},
                include=["metadatas"]
            )
        else:
            # Global query — fetch everything (used by /api/status for the count)
            collection_data = store._collection.get(include=["metadatas"])

        metadatas = collection_data.get("metadatas", []) if collection_data else []

        unique_sources = set()
        for meta in metadatas:
            if meta and "source" in meta:
                # When scoped, skip any stray "NO_SESSION_DEFINED" docs
                if session_id and meta.get("session_id") != session_id:
                    continue
                unique_sources.add(meta["source"])

        return sorted(list(unique_sources))
    except Exception:
        return []
