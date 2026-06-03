import os
import re
import shutil
import uuid
import datetime
import logging
import requests
import wave
import io
import urllib.request
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

# Import app components
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
from graph import run_graph
from memory import (
    ChatMessageModel,
    ChatSessionModel,
    get_all_sessions,
    update_session_title,
    delete_session,
)
from middleware import RequestLoggingMiddleware, setup_cors


def _parse_retry_after(error_str: str) -> int:
    """Parse Groq rate-limit retry delay from error detail string.

    Groq error messages look like:
      'Please try again in 2s.'
      'Please try again in 1m30s.'
      'Please try again in 1m.'
    Returns seconds as an int (default 60 when not parsable).
    """
    s = (error_str or "").lower()
    # e.g. "1m30s"
    m = re.search(r'try again in (\d+)m(\d+)s', s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    # e.g. "1m"
    m = re.search(r'try again in (\d+)m', s)
    if m:
        return int(m.group(1)) * 60
    # e.g. "2s" or "2.5s"
    m = re.search(r'try again in ([\d.]+)s', s)
    if m:
        return max(1, int(float(m.group(1))))
    return 60  # safe default


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
    extra: dict = {}

    if isinstance(exc, InvalidDocumentError):
        status_code = 400
    elif isinstance(exc, LLMProviderError):
        if exc.error_code == "GROQ_RATE_LIMIT":
            status_code = 429
            # Parse retry delay from Groq error message and send to frontend
            extra["retry_after_seconds"] = _parse_retry_after(exc.details or "")
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
            "details": exc.details,
            **extra,          # includes retry_after_seconds on 429
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


class TTSRequest(BaseModel):
    text: str


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
    Multi-Agent RAG endpoint. Routes the query through the LangGraph pipeline:
    Supervisor -> Query Planner -> Memory -> Retrieval -> Web Research -> Synthesis -> Critique
    """
    # 1. API key verification
    if not settings.GROQ_API_KEY or "your_groq_api" in settings.GROQ_API_KEY.lower():
        raise LLMProviderError(
            message="Groq API Key is missing. Please configure GROQ_API_KEY in the .env file.",
            error_code="GROQ_API_KEY_MISSING"
        )

    # 2. Update session title if still default
    session = db.query(ChatSessionModel).filter_by(session_id=req.session_id).first()
    if session and (session.title == "New Conversation" or session.title == req.session_id):
        new_title = req.message.strip()
        if len(new_title) > 40:
            new_title = new_title[:37] + "..."
        session.title = new_title
        db.commit()

    current_time = datetime.datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")

    try:
        # 3. Save human message to MySQL
        human_msg = ChatMessageModel(
            session_id=req.session_id,
            role="human",
            content=req.message
        )
        db.add(human_msg)
        db.commit()

        # 4. Run the multi-agent graph
        result = run_graph(
            raw_query=req.message,
            session_id=req.session_id,
            current_time=current_time,
            db_session=db,
            model=req.model,
        )

        final_answer = result["final_answer"]
        citations    = result["citations"]
        agent_trace  = result["agent_trace"]
        confidence   = result["critique_score"]

        # 5. Save AI response to MySQL
        ai_msg = ChatMessageModel(
            session_id=req.session_id,
            role="ai",
            content=final_answer,
            citations=citations
        )
        db.add(ai_msg)
        db.commit()

        return {
            "response":        final_answer,
            "citations":       citations,
            "session_id":      req.session_id,
            "agent_trace":     agent_trace,
            "confidence":      round(confidence, 2),
            "sub_queries":     result.get("sub_queries_used", []),
        }

    except RAGAgentError:
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
        elif "authentication" in err_lower or "api_key" in err_lower or "401" in err_lower:
            raise LLMProviderError(
                message="Authentication with Groq API failed. Please check your GROQ_API_KEY.",
                details=err_msg,
                error_code="GROQ_AUTH_FAILURE"
            )
        elif "connection" in err_lower or "timeout" in err_lower:
            raise LLMProviderError(
                message="Failed to connect to Groq API. Check your internet connection.",
                details=err_msg,
                error_code="GROQ_CONNECTION_ERROR"
            )
        elif "database" in err_lower or "sqlalchemy" in err_lower or "mysql" in err_lower:
            raise DatabaseConnectionError(
                message="A database error occurred while processing the chat session.",
                details=err_msg
            )
        else:
            logger.error(f"Error in multi-agent chat: {str(e)}", exc_info=True)
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
async def status_endpoint(session_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Check status of connected services (MySQL, Chroma).
    
    Pass ?session_id=<id> to get the indexed document count for the active session.
    Without session_id the count reflects all documents across all sessions.
    """
    mysql_ok = False
    try:
        db.execute(text("SELECT 1"))
        mysql_ok = True
    except Exception:
        pass
        
    chroma_ok = False
    indexed_files_count = 0
    try:
        # Pass session_id so the counter is scoped to the current session when provided.
        # get_indexed_files(None) now correctly returns ALL docs for the global view.
        indexed_files = get_indexed_files(session_id=session_id if session_id else None)
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

@app.post("/api/stt")
async def speech_to_text_endpoint(
    file: UploadFile = File(...),
    engine: Optional[str] = Form("auto")
):
    """
    Speech-To-Text transcription endpoint.
    Accepts audio file and transcribes it using Groq Whisper or custom-deployed local STT.
    Automatically falls back to local STT if Groq fails or is offline (when engine is 'auto').
    """
    try:
        audio_content = await file.read()
        filename = file.filename or "audio.webm"
    except Exception as e:
        logger.error(f"Failed to read uploaded audio file: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded audio file.")

    used_engine = None
    transcript = None
    errors = []

    # 1. Groq Whisper Path (Online)
    if engine in ("auto", "groq"):
        if not settings.GROQ_API_KEY or "your_groq_api" in settings.GROQ_API_KEY.lower():
            err_msg = "Groq API Key is not configured."
            logger.warning(err_msg)
            errors.append(err_msg)
        else:
            try:
                headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
                data = {
                    "model": "whisper-large-v3",
                    "language": "en"
                }
                
                # Make HTTP POST request to Groq API
                response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files={"file": (filename, audio_content, file.content_type or "audio/webm")},
                    data=data,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    res_json = response.json()
                    transcript = res_json.get("text", "").strip()
                    if transcript:
                        used_engine = "groq"
                else:
                    err_msg = f"Groq Whisper API error (status {response.status_code}): {response.text}"
                    logger.warning(err_msg)
                    errors.append(err_msg)
            except Exception as e:
                err_msg = f"Failed to connect to Groq Whisper API (system might be offline): {str(e)}"
                logger.warning(err_msg)
                errors.append(err_msg)

    # 2. Custom STT / Local Piper Path (Offline Fallback or Explicit)
    if not transcript and (engine == "custom" or (engine == "auto" and errors)):
        if not settings.CUSTOM_STT_URL:
            err_msg = "Custom/Local STT URL is not configured."
            logger.error(err_msg)
            errors.append(err_msg)
        else:
            try:
                logger.info(f"Attempting offline/local transcription via Custom STT (Piper) at {settings.CUSTOM_STT_URL}...")
                
                # Forward audio file to user's locally deployed STT service (e.g. Piper)
                custom_response = requests.post(
                    settings.CUSTOM_STT_URL,
                    files={"file": (filename, audio_content, file.content_type or "audio/webm")},
                    timeout=15.0
                )
                
                if custom_response.status_code == 200:
                    res_json = custom_response.json()
                    # Support multiple potential return formats (text, transcript, transcription, etc.)
                    if isinstance(res_json, str):
                        transcript = res_json
                    elif isinstance(res_json, dict):
                        transcript = (
                            res_json.get("text") or 
                            res_json.get("transcript") or 
                            res_json.get("transcription") or 
                            res_json.get("text_transcription")
                        )
                        if isinstance(transcript, dict):
                            transcript = transcript.get("text") or str(transcript)
                    
                    if transcript:
                        transcript = transcript.strip()
                        used_engine = "custom"
                else:
                    err_msg = f"Custom STT service error (status {custom_response.status_code}): {custom_response.text}"
                    logger.error(err_msg)
                    errors.append(err_msg)
            except Exception as e:
                err_msg = f"Failed to connect to Custom STT service: {str(e)}"
                logger.error(err_msg)
                errors.append(err_msg)

    if transcript:
        logger.info(f"Successfully transcribed audio using engine: {used_engine}")
        return {
            "status": "success",
            "text": transcript,
            "engine": used_engine,
            "fallback_occurred": (engine == "auto" and used_engine == "custom")
        }
    else:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to transcribe audio. All attempted engines failed.",
                "errors": errors
            }
        )

# Lazy loading of Piper TTS Voice
tts_voice = None

def get_tts_voice():
    global tts_voice
    if tts_voice is None:
        try:
            from piper import PiperVoice
        except ImportError:
            logger.error("piper-tts is not installed. Make sure to run pip install -r requirements-stt.txt")
            raise HTTPException(
                status_code=500,
                detail="Piper TTS library is not installed in the environment."
            )

        model_path = os.path.abspath(settings.TTS_VOICE_MODEL)
        config_path = os.path.abspath(settings.TTS_VOICE_CONFIG)
        
        # Automatic download from Hugging Face if files are missing
        voice_name = "en_US-lessac-medium"
        model_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{voice_name}.onnx"
        config_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{voice_name}.onnx.json"
        
        try:
            if not os.path.exists(model_path):
                logger.info(f"Downloading TTS model from {model_url} to {model_path}...")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                urllib.request.urlretrieve(model_url, model_path)
                logger.info("TTS Model downloaded successfully.")
                
            if not os.path.exists(config_path):
                logger.info(f"Downloading TTS config from {config_url} to {config_path}...")
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                urllib.request.urlretrieve(config_url, config_path)
                logger.info("TTS Config downloaded successfully.")
                
            logger.info(f"Loading Piper voice model from {model_path}...")
            tts_voice = PiperVoice.load(model_path, config_path=config_path)
            logger.info("Piper voice model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load or download Piper TTS voice: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load Piper voice model: {str(e)}"
            )
            
    return tts_voice

@app.post("/api/tts")
async def text_to_speech_endpoint(req: TTSRequest):
    """
    Synthesize text to speech using local Piper neural engine.
    Returns wav audio stream.
    """
    text_to_speak = req.text.strip()
    if not text_to_speak:
        raise HTTPException(status_code=400, detail="Text for speech synthesis cannot be empty.")
        
    try:
        voice = get_tts_voice()
        
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            voice.synthesize_wav(text_to_speak, wav_file)
            
        wav_io.seek(0)
        return StreamingResponse(wav_io, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech synthesis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to synthesize speech: {str(e)}"
        )

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
