from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from typing import Optional
from qdrant_client import QdrantClient

class Settings(BaseSettings):
    """Application settings."""
    openai_api_key: str
    qdrant_url: str
    qdrant_api_key: str

    @property
    def qdrant_client(self) -> QdrantClient:
        return QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load OpenAI API key from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Load Qdrant URL from environment
        self.qdrant_url = os.getenv("QDRANT_URL")
        if not self.qdrant_url:
            raise ValueError("QDRANT_URL environment variable is required")
        
        # Load Qdrant API key from environment
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY environment variable is required")

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings() 