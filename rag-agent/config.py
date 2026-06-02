import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Pydantic Settings reads variables from .env file or environment
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server Configuration
    HOST: str = Field(default="127.0.0.1")
    PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=True)

    # API Keys & LLM settings
    GROQ_API_KEY: str = Field(default="")
    LLM_MODEL: str = Field(default="llama-3.1-8b-instant")
    LLM_TEMPERATURE: float = Field(default=0.2)

    # Speech-To-Text (STT) Settings
    DEFAULT_STT_ENGINE: str = Field(default="auto")  # "auto", "groq", "custom"
    CUSTOM_STT_URL: str = Field(default="http://127.0.0.1:8001/transcribe")  # Custom/Piper STT endpoint

    # MySQL Configuration
    MYSQL_HOST: str = Field(default="localhost")
    MYSQL_PORT: int = Field(default=3306)
    MYSQL_USER: str = Field(default="root")
    MYSQL_PASSWORD: str = Field(default="")
    MYSQL_DATABASE: str = Field(default="rag_agent")

    # Directories
    CHROMA_DB_PATH: str = Field(default="./chroma_db")
    DATA_DIR: str = Field(default="./data")

    @property
    def mysql_url(self) -> str:
        # Construct pymysql URL for SQLAlchemy
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"

# Global config instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
