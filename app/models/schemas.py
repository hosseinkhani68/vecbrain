from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DocumentBase(BaseModel):
    """Base document model."""
    text: str = Field(..., description="The text content of the document")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata for the document")

class DocumentCreate(BaseModel):
    """Model for creating a new document."""
    text: str
    metadata: Optional[Dict[str, Any]] = {}

class DocumentResponse(BaseModel):
    """Model for document response."""
    id: str = Field(..., description="Unique identifier for the document")
    text: str
    metadata: Dict[str, Any]
    embedding: List[float] = Field(..., description="Vector embedding of the document")

class SearchQuery(BaseModel):
    """Model for search queries."""
    text: str = Field(..., description="The search query text")
    limit: int = 5

class SearchResult(BaseModel):
    """Model for search results."""
    text: str = Field(..., description="The text content of the found document")
    score: float = Field(..., description="Similarity score of the match")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata of the document")

class SearchResponse(BaseModel):
    """Model for search response."""
    results: List[Dict[str, Any]] = Field(..., description="List of search results")

class QuestionResponse(BaseModel):
    """Model for question answering response."""
    answer: str = Field(..., description="The generated answer to the question")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used to generate the answer")

class SimplifyRequest(BaseModel):
    text: str

class ChatMessage(BaseModel):
    """Model for chat messages."""
    id: str = Field(..., description="Unique identifier for the chat message")
    text: str = Field(..., description="The message text")
    role: str = Field(..., description="The role of the message sender (user/assistant)")
    timestamp: str = Field(..., description="Timestamp of the message")

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    text: str = Field(..., description="The message text to process")
    context_id: Optional[str] = Field(None, description="Optional context ID for conversation continuity")

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    id: str = Field(..., description="Unique identifier for the message")
    text: str = Field(..., description="The response text")
    role: str = Field(..., description="The role of the message sender (user/assistant)")
    timestamp: str = Field(..., description="Timestamp of the message")
    context_id: str = Field(..., description="Context ID for conversation grouping")

class ChatHistoryResponse(BaseModel):
    """Response model for chat history endpoint."""
    id: str = Field(..., description="Unique identifier for the message")
    text: str = Field(..., description="The message text")
    role: str = Field(..., description="The role of the message sender (user/assistant)")
    timestamp: str = Field(..., description="Timestamp of the message") 