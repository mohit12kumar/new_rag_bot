class RAGAgentError(Exception):
    """Base exception for all RAG Agent errors."""
    def __init__(self, message: str, details: str = None, error_code: str = "INTERNAL_ERROR"):
        super().__init__(message)
        self.message = message
        self.details = details
        self.error_code = error_code

class DatabaseConnectionError(RAGAgentError):
    """Exception raised when database connection or operations fail."""
    def __init__(self, message: str, details: str = None):
        super().__init__(message, details, "DATABASE_ERROR")

class LLMProviderError(RAGAgentError):
    """Exception raised when Groq client fails (rate limits, auth, connection, etc.)."""
    def __init__(self, message: str, details: str = None, error_code: str = "GROQ_ERROR"):
        super().__init__(message, details, error_code)

class VectorStoreError(RAGAgentError):
    """Exception raised when Chroma operations fail."""
    def __init__(self, message: str, details: str = None):
        super().__init__(message, details, "VECTOR_STORE_ERROR")

class InvalidDocumentError(RAGAgentError):
    """Exception raised when document format is unsupported or parsing/split fails."""
    def __init__(self, message: str, details: str = None):
        super().__init__(message, details, "INVALID_DOCUMENT_ERROR")
