from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class DocumentBase(BaseModel):
    """Base document model."""
    text: str = Field(..., description="The text content of the document")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata for the document")

class DocumentCreate(DocumentBase):
    """Model for creating a new document."""
    pass

class DocumentResponse(DocumentBase):
    """Model for document response."""
    id: int = Field(..., description="Unique identifier for the document")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding of the document")

class SearchQuery(BaseModel):
    """Model for search queries."""
    text: str = Field(..., description="The search query text")
    limit: Optional[int] = Field(default=5, description="Maximum number of results to return")

class SearchResult(BaseModel):
    """Model for search results."""
    text: str = Field(..., description="The text content of the found document")
    score: float = Field(..., description="Similarity score of the match")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata of the document")

class SearchResponse(BaseModel):
    """Model for search response."""
    results: List[SearchResult] = Field(..., description="List of search results")

class QuestionResponse(BaseModel):
    """Model for question answering response."""
    answer: str = Field(..., description="The generated answer to the question")
    sources: List[SearchResult] = Field(..., description="Source documents used to generate the answer") 