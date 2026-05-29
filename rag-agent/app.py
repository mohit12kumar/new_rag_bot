import os
import shutil
import uuid
import datetime
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

from config import settings
from database import get_db
from exceptions import (
    RAGAgentError,
    DatabaseConnectionError,
    LLMProviderError,
    VectorStoreError,
    InvalidDocumentError
)
from rag import ingest_file, delete_file_from_store, get_indexed_files
from tools import retrieved_citations, current_session_id
from agent import get_agent_executor
from memory import (
    ChatMessageModel,
    ChatSessionModel,
    get_all_sessions,
    update_session_title,
    delete_session,
)
from middleware import RequestLoggingMiddleware, setup_cors

# Configure logging
logger = logging.getLogger("rag_agent_app")

app = FastAPI(
    title="Antigravity RAG Agent",
    description="LangChain RAG Agent with MySQL & Chroma DB Backend",
    version="1.0.0"
)

# Setup CORS & Logging Middleware
setup_cors(app)
app.add_middleware(RequestLoggingMiddleware)

# Custom Global Exception Handlers
@app.exception_handler(RAGAgentError)
async def rag_agent_exception_handler(request: Request, exc: RAGAgentError):
    status_code = 500
    if isinstance(exc, InvalidDocumentError):
        status_code = 400
    elif isinstance(exc, LLMProviderError):
        if exc.error_code == "GROQ_RATE_LIMIT":
            status_code = 429
        elif exc.error_code == "GROQ_AUTH_FAILURE":
            status_code = 401
        elif exc.error_code == "GROQ_API_KEY_MISSING":
            status_code = 400
        else:
            status_code = 502
    elif isinstance(exc, DatabaseConnectionError):
        status_code = 503
    elif isinstance(exc, VectorStoreError):
        status_code = 500

    logger.error(
        f"RAGAgentError ({exc.error_code}) during {request.method} {request.url.path}: {exc.message}. Details: {exc.details or ''}"
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "details": exc.details
        }
    )

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.error(f"Database operational error during {request.method} {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "The database service is temporarily unavailable or misconfigured. Please check if your MySQL server is running.",
            "error_code": "DATABASE_CONNECTION_ERROR",
            "details": str(exc)
        }
    )

# Request Models
class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: Optional[str] = None



class SessionTitleRequest(BaseModel):
    title: str

# ----------------- API Endpoints -----------------

@app.get("/api/sessions")
async def list_sessions_endpoint(db: Session = Depends(get_db)):
    """Retrieve all conversations for the sidebar."""
    return get_all_sessions(db)

@app.post("/api/sessions")
async def create_session_endpoint(title: Optional[str] = None, db: Session = Depends(get_db)):
    """Initialize a new conversation thread."""
    session_id = str(uuid.uuid4())
    new_session = ChatSessionModel(
        session_id=session_id,
        title=title or "New Conversation"
    )
    db.add(new_session)
    db.commit()
    return {"session_id": session_id, "title": new_session.title}

@app.put("/api/sessions/{session_id}")
async def rename_session_endpoint(session_id: str, req: SessionTitleRequest, db: Session = Depends(get_db)):
    """Rename a conversation thread."""
    success = update_session_title(db, session_id, req.title)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success"}

@app.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, db: Session = Depends(get_db)):
    """Delete a conversation thread and its messages."""
    success = delete_session(db, session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success"}

@app.get("/api/sessions/{session_id}/messages")
async def get_messages_endpoint(session_id: str, db: Session = Depends(get_db)):
    """Get chat logs for a specific session."""
    messages = (
        db.query(ChatMessageModel)
        .filter_by(session_id=session_id)
        .order_by(ChatMessageModel.id.asc())
        .all()
    )
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "citations": m.citations or [],
            "created_at": m.created_at.isoformat()
        }
        for m in messages
    ]

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Submits user query to the LangChain agent, triggers retrieval 
    if tools are called, saves output, and returns response with citations.
    """
    # 1. API key verification for Groq
    if not settings.GROQ_API_KEY or "your_groq_api" in settings.GROQ_API_KEY.lower():
        raise LLMProviderError(
            message="Groq API Key is missing. Please configure GROQ_API_KEY in the .env file.",
            error_code="GROQ_API_KEY_MISSING"
        )

    # 2. Reset context variable for citations and session
    retrieved_citations.set([])
    current_session_id.set(req.session_id)

    # 2.5 Update session title if it is default ("New Conversation" or UUID)
    session = db.query(ChatSessionModel).filter_by(session_id=req.session_id).first()
    if session and (session.title == "New Conversation" or session.title == req.session_id):
        new_title = req.message.strip()
        if len(new_title) > 40:
            new_title = new_title[:37] + "..."
        session.title = new_title
        db.commit()

    current_time = datetime.datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
    
    try:
        # 3. Retrieve agent with dynamic model
        agent_executor = get_agent_executor(db, model=req.model)

        # 4. Invoke agent (RunnableWithMessageHistory saves input/output automatically)
        response = agent_executor.invoke(
            {"input": req.message, "current_time": current_time},
            config={"configurable": {"session_id": req.session_id}}
        )

        # 5. Extract citations captured from the thread run
        citations = retrieved_citations.get()

        # 6. Save citations into MySQL db for the AI message
        if citations:
            last_msg = (
                db.query(ChatMessageModel)
                .filter_by(session_id=req.session_id)
                .order_by(ChatMessageModel.id.desc())
                .first()
            )
            if last_msg and last_msg.role == "ai":
                last_msg.citations = citations
                db.commit()

        return {
            "response": response["output"],
            "citations": citations,
            "session_id": req.session_id
        }

    except RAGAgentError:
        # Re-raise custom exceptions so global exception handlers process them correctly
        raise
    except Exception as e:
        err_msg = str(e)
        err_lower = err_msg.lower()
        if "rate_limit_exceeded" in err_lower or "429" in err_lower or "rate limit" in err_lower:
            raise LLMProviderError(
                message="Groq API rate limit exceeded. Please wait a moment before trying again.",
                details=err_msg,
                error_code="GROQ_RATE_LIMIT"
            )
        elif "authentication" in err_lower or "api_key" in err_lower or "invalid_api_key" in err_lower or "401" in err_lower:
            raise LLMProviderError(
                message="Authentication with Groq API failed. Please check that your GROQ_API_KEY in the .env file is correct and active.",
                details=err_msg,
                error_code="GROQ_AUTH_FAILURE"
            )
        elif "connection" in err_lower or "timeout" in err_lower or "connect" in err_lower:
            raise LLMProviderError(
                message="Failed to connect to Groq API. Please verify your internet connection or try again later.",
                details=err_msg,
                error_code="GROQ_CONNECTION_ERROR"
            )
        elif "failed to call a function" in err_lower or "failed_generation" in err_lower or "tool_use_failed" in err_lower:
            raise LLMProviderError(
                message="The language model failed to format its tool call correctly. This is a transient Groq formatting issue. Please try asking your question again or rephrase it slightly.",
                details=err_msg,
                error_code="GROQ_TOOL_CALL_FAILED"
            )
        elif "database" in err_lower or "sqlalchemy" in err_lower or "mysql" in err_lower:
            raise DatabaseConnectionError(
                message="A database error occurred while processing the chat session.",
                details=err_msg
            )
        else:
            logger.error(f"Error in chat execution: {str(e)}", exc_info=True)
            raise RAGAgentError(
                message=f"An error occurred during chat execution: {err_msg}",
                details=err_msg,
                error_code="CHAT_EXECUTION_ERROR"
            )

@app.post("/api/upload")
async def upload_file_endpoint(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload a document file (PDF, TXT, MD), saves it locally in a session-specific directory, 
    and ingests it into Chroma Vector database.
    """
    if not session_id or session_id.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Session ID is required to upload files. Anonymous or global uploads are not permitted."
        )

    filename = file.filename
    # Clean filename slightly
    filename = os.path.basename(filename)
    
    # Isolate files on disk by session_id to prevent collision
    session_dir = os.path.join(settings.DATA_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    save_path = os.path.join(session_dir, filename)

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Ingest into Chroma Vector Store with session context
        chunks_count = ingest_file(save_path, session_id=session_id)
        
        return {
            "filename": filename,
            "status": "success",
            "chunks": chunks_count,
            "message": f"Successfully parsed and split into {chunks_count} chunks."
        }
    except RAGAgentError:
        # Cleanup file if ingestion failed
        if os.path.exists(save_path):
            os.remove(save_path)
        raise
    except Exception as e:
        # Cleanup file if ingestion failed
        if os.path.exists(save_path):
            os.remove(save_path)
        logger.error(f"Error during file upload/ingestion: {str(e)}", exc_info=True)
        raise InvalidDocumentError(
            message=f"An unexpected error occurred during upload/ingestion of '{filename}'.",
            details=str(e)
        )

@app.get("/api/files")
async def list_files_endpoint(session_id: Optional[str] = None):
    """Retrieve list of local files for the current session and whether they are indexed in Chroma."""
    if not session_id or session_id.strip() == "":
        # Strictly return empty list to protect privacy and prevent cross-session document leakage
        return []

    try:
        session_dir = os.path.join(settings.DATA_DIR, session_id)
        if not os.path.exists(session_dir):
            return []
            
        files = os.listdir(session_dir)
        indexed = get_indexed_files(session_id=session_id)
        
        result = []
        for f in files:
            file_path = os.path.join(session_dir, f)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                result.append({
                    "filename": f,
                    "size_bytes": stat.st_size,
                    "created_at": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "status": "Indexed" if f in indexed else "Not Indexed"
                })
        return result
    except Exception as e:
        raise VectorStoreError(
            message="Failed to list files or query the indexed documents database.",
            details=str(e)
        )

@app.delete("/api/files/{filename}")
async def delete_file_endpoint(filename: str, session_id: Optional[str] = None):
    """Delete a file from the server and remove its indexes from Chroma."""
    if not session_id or session_id.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Session ID is required to delete files."
        )

    session_dir = os.path.join(settings.DATA_DIR, session_id)
    file_path = os.path.join(session_dir, filename)
    
    # Delete from disk
    if os.path.exists(file_path):
        os.remove(file_path)
        
    # Delete from Chroma
    delete_file_from_store(filename, session_id=session_id)
    
    return {"status": "success", "message": f"{filename} deleted successfully."}

@app.get("/api/status")
async def status_endpoint(db: Session = Depends(get_db)):
    """Check status of connected services (MySQL, Chroma)."""
    mysql_ok = False
    try:
        db.execute(text("SELECT 1"))
        mysql_ok = True
    except Exception:
        pass
        
    chroma_ok = False
    indexed_files_count = 0
    try:
        indexed_files = get_indexed_files()
        indexed_files_count = len(indexed_files)
        chroma_ok = True
    except Exception:
        pass
        
    return {
        "status": "online" if (mysql_ok and chroma_ok) else "degraded",
        "mysql": "connected" if mysql_ok else "disconnected",
        "chroma": "connected" if chroma_ok else "disconnected",
        "indexed_documents": indexed_files_count
    }

# ----------------- Premium Single Page UI -----------------

@app.get("/", response_class=FileResponse)
async def serve_ui():
    """Serves the premium single-page UI."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "templates", "index.html"))

# ----------------- Main Launcher -----------------

if __name__ == "__main__":
    import uvicorn
    # Pre-create directory folders
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
    
    logger.info(f"Starting Antigravity RAG Agent server on {settings.HOST}:{settings.PORT}")
    uvicorn.run("app:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
