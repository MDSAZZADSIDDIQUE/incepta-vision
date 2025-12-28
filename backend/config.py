"""
Configuration management for Incepta Vision backend.
Centralizes all environment variables and application settings.
"""
import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration class."""
    
    # Database Configuration
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "1521"))
    DB_SERVICE: str = os.getenv("DB_SERVICE", "")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    
    # Oracle Client
    ORACLE_HOME: Optional[str] = os.getenv("ORACLE_HOME")
    
    # LLM Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen-2.5-32b")
    
    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Query Limits
    MAX_ROWS: int = int(os.getenv("MAX_ROWS", "2000"))
    SQL_RETRY_ATTEMPTS: int = int(os.getenv("SQL_RETRY_ATTEMPTS", "3"))
    
    # Dictionary Path
    DICTIONARY_PATH: str = os.getenv("DICTIONARY_PATH", "")
    
    # CORS Settings
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Security
    FORBIDDEN_SQL_KEYWORDS: List[str] = [
        "DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", 
        "ALTER", "GRANT", "REVOKE", "CREATE", "REPLACE", "MERGE"
    ]
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        required_fields = ["DB_HOST", "DB_SERVICE", "DB_USER", "DB_PASSWORD"]
        missing = [field for field in required_fields if not getattr(cls, field)]
        
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        
        if not cls.GROQ_API_KEY:
            import logging
            logging.warning("GROQ_API_KEY is not set. /chat endpoint will fail.")


# Singleton instance
config = Config()
