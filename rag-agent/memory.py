import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Session
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from database import Base, engine

class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"
    
    session_id = Column(String(255), primary_key=True)
    title = Column(String(255), nullable=True, default="New Conversation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)  # 'human', 'ai', 'system'
    content = Column(Text(length=16777215), nullable=False)  # MEDIUMTEXT to allow long messages
    citations = Column(JSON, nullable=True)  # List of dictionaries: [{"file": "...", "snippet": "..."}]
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create the tables in MySQL automatically
Base.metadata.create_all(bind=engine)

class MySQLChatMessageHistory(BaseChatMessageHistory):
    """
    MySQL-backed chat message history for LangChain agents.
    """
    def __init__(self, session_id: str, db_session: Session):
        self.session_id = session_id
        self.db = db_session
        self._ensure_session_exists()

    def _ensure_session_exists(self):
        """
        Verify that the session exists in the sessions table; if not, create it.
        """
        try:
            session = self.db.query(ChatSessionModel).filter_by(session_id=self.session_id).first()
            if not session:
                new_session = ChatSessionModel(session_id=self.session_id, title="New Conversation")
                self.db.add(new_session)
                self.db.commit()
        except Exception as e:
            self.db.rollback()
            print(f"Warning: Failed to verify/create chat session {self.session_id}: {e}")

    @property
    def messages(self) -> List[BaseMessage]:
        """
        Retrieve messages from MySQL and format them as LangChain BaseMessage objects.
        """
        db_messages = (
            self.db.query(ChatMessageModel)
            .filter_by(session_id=self.session_id)
            .order_by(ChatMessageModel.id.asc())
            .all()
        )
        
        messages = []
        for msg in db_messages:
            if msg.role == "human":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "ai":
                # Reconstruct AIMessage and carry citations inside additional_kwargs
                add_kwargs = {}
                if msg.citations:
                    add_kwargs["citations"] = msg.citations
                messages.append(AIMessage(content=msg.content, additional_kwargs=add_kwargs))
            elif msg.role == "system":
                messages.append(SystemMessage(content=msg.content))
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """
        Append a message to the database.
        """
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "human"

        citations = message.additional_kwargs.get("citations") if hasattr(message, "additional_kwargs") else None

        db_msg = ChatMessageModel(
            session_id=self.session_id,
            role=role,
            content=message.content,
            citations=citations
        )
        self.db.add(db_msg)
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            print(f"Error saving chat message: {e}")
            raise e

    def add_ai_message_with_citations(self, content: str, citations: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Utility method to insert an AI response with citations directly.
        """
        db_msg = ChatMessageModel(
            session_id=self.session_id,
            role="ai",
            content=content,
            citations=citations
        )
        self.db.add(db_msg)
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            print(f"Error saving AI message with citations: {e}")
            raise e

    def clear(self) -> None:
        """
        Clear all messages for this session.
        """
        try:
            self.db.query(ChatMessageModel).filter_by(session_id=self.session_id).delete()
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            print(f"Error clearing chat messages: {e}")
            raise e


def get_all_sessions(db: Session) -> List[Dict[str, Any]]:
    """
    Retrieve all conversation sessions from the database for sidebar lists.
    """
    sessions = db.query(ChatSessionModel).order_by(ChatSessionModel.created_at.desc()).all()
    return [
        {
            "session_id": s.session_id,
            "title": s.title,
            "created_at": s.created_at.isoformat()
        }
        for s in sessions
    ]

def update_session_title(db: Session, session_id: str, new_title: str) -> bool:
    """
    Update the title of a conversation thread.
    """
    try:
        session = db.query(ChatSessionModel).filter_by(session_id=session_id).first()
        if session:
            session.title = new_title
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        print(f"Error updating session title: {e}")
        return False

def delete_session(db: Session, session_id: str) -> bool:
    """
    Delete a session and all its messages.
    """
    try:
        session = db.query(ChatSessionModel).filter_by(session_id=session_id).first()
        if session:
            db.delete(session)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        print(f"Error deleting session: {e}")
        return False
