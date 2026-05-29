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


def get_vector_store() -> Chroma:
    """
    Initialize and return the Chroma vector store instance.
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
    Perform a similarity search in Chroma and format the results.
    
    Returns a list of structured results containing:
      - content: Text snippet
      - source: Filename
      - page: Page number (if available)
      - score: Distance score
    """
    try:
        store = get_vector_store()
        # Strictly enforce search filtering by session_id to protect other sessions
        active_session = session_id if session_id else "NO_SESSION_DEFINED"
        search_filter = {"session_id": active_session}
        # Search with score (returns tuple of (Document, score))
        # Chroma returns L2 distance; lower is closer (more similar)
        raw_results = store.similarity_search_with_score(query, k=k, filter=search_filter)
    except Exception as e:
        raise VectorStoreError(
            message="Failed to perform query search in the vector database.",
            details=str(e)
        )
    
    try:
        formatted_results = []
        for doc, score in raw_results:
            formatted_results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "score": float(score)
            })
        return formatted_results
    except Exception as e:
        raise VectorStoreError(
            message="Failed to process search results from the vector database.",
            details=str(e)
        )

def get_indexed_files(session_id: str = None) -> List[str]:
    """
    Retrieve list of unique filenames that are currently indexed.
    """
    try:
        store = get_vector_store()
        # Fetch metadata from all items in the collection
        collection_data = store._collection.get(include=["metadatas"])
        metadatas = collection_data.get("metadatas", [])
        
        # Extract unique sources
        unique_sources = set()
        # Strictly require a valid session_id to list files
        target_session = session_id if session_id else "NO_SESSION_DEFINED"
        for meta in metadatas:
            if meta and "source" in meta:
                if meta.get("session_id") != target_session:
                    continue
                unique_sources.add(meta["source"])
                
        return sorted(list(unique_sources))
    except Exception:
        return []
