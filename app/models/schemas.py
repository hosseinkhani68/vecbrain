from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Document Models
class Document(BaseModel):
    """Schema for a document."""
    doc_id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Title of the document")
    source: str = Field(..., description="Source/path of the document")
    chunks: int = Field(..., description="Number of chunks in the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

class DocumentChunk(BaseModel):
    """Schema for a document chunk."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    doc_id: str = Field(..., description="ID of the parent document")
    text: str = Field(..., description="Content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

class DocumentCreate(BaseModel):
    """Model for creating a new document."""
    text: str = Field(..., description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class DocumentResponse(BaseModel):
    """Model for document response."""
    id: str = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="Document text content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    embedding: List[float] = Field(..., description="Vector embedding of the document")

class DocumentProcessRequest(BaseModel):
    """Request model for document processing."""
    file_path: str = Field(..., description="Path to the document file")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata for the document")

class DocumentProcessResponse(BaseModel):
    """Response model for document processing."""
    doc_id: str = Field(..., description="Unique identifier for the processed document")
    chunks: int = Field(..., description="Number of chunks the document was split into")
    source: str = Field(..., description="Source file path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")

class DocumentSearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=5, description="Maximum number of results to return")

class DocumentSearchResponse(BaseModel):
    """Response model for document search."""
    results: List[Dict[str, Any]] = Field(..., description="Search results with text and metadata")
    total: int = Field(..., description="Total number of results found")

class DocumentChunksRequest(BaseModel):
    """Request model for retrieving document chunks."""
    doc_id: str = Field(..., description="Document ID to retrieve chunks for")

class DocumentChunksResponse(BaseModel):
    """Response model for document chunks."""
    chunks: List[Dict[str, Any]] = Field(..., description="Document chunks with text and metadata")

# Chat Models
class ChatMessage(BaseModel):
    """Schema for a chat message."""
    id: str = Field(..., description="Unique identifier for the message")
    text: str = Field(..., description="Content of the message")
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    context_id: Optional[str] = Field(None, description="Context ID for conversation grouping")

class ChatRequest(BaseModel):
    """Schema for a chat request."""
    text: str = Field(..., description="Message text", max_length=4000)
    context_id: Optional[str] = Field(None, description="Context ID for conversation grouping")
    history: List[ChatMessage] = Field(default_factory=list, description="Chat history")
    stream: bool = Field(default=False, description="Whether to stream the response")

class ChatResponse(BaseModel):
    """Schema for a chat response."""
    id: str = Field(..., description="Unique identifier for the response")
    text: str = Field(..., description="Response text")
    role: str = Field(default="assistant", description="Role of the response")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    context_id: Optional[str] = Field(None, description="Context ID for conversation grouping")
    history: List[ChatMessage] = Field(default_factory=list, description="Updated chat history")

class ChatHistoryResponse(BaseModel):
    """Response model for chat history endpoint."""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")

# Agent Models
class AgentQueryRequest(BaseModel):
    """Request model for agent queries."""
    query: str = Field(..., description="The query to process using the agent")

class AgentQueryResponse(BaseModel):
    """Response model for agent queries."""
    response: str = Field(..., description="The agent's response")
    tools_used: List[str] = Field(default_factory=list, description="List of tools used by the agent")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the response")
    error: Optional[str] = Field(None, description="Error message if any")

# Prompt Models
class PromptRequest(BaseModel):
    """Schema for a prompt generation request."""
    template_name: str = Field(..., description="Name of the prompt template to use")
    input_data: Dict[str, Any] = Field(..., description="Input data for the template")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    history: Optional[List[Dict[str, str]]] = Field(None, description="Conversation history")

class PromptResponse(BaseModel):
    """Schema for a prompt generation response."""
    response: str = Field(..., description="Generated response")
    template_used: str = Field(..., description="Name of the template used")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    error: Optional[str] = Field(None, description="Error message if generation failed")

class TemplateInfoResponse(BaseModel):
    """Schema for template information."""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    input_variables: List[str] = Field(..., description="Required input variables")

# Streaming Models
class StreamResponse(BaseModel):
    """Model for streaming response chunks."""
    chunk: str = Field(..., description="A chunk of the streaming response")
    done: bool = Field(default=False, description="Whether this is the final chunk")
    error: Optional[str] = Field(None, description="Error message if any")

class SearchQuery(BaseModel):
    """Model for search queries."""
    text: str = Field(..., description="The search query text")
    limit: int = 5

class SearchResult(BaseModel):
    """Model for search results."""
    text: str = Field(..., description="The text content of the found document")
    score: float = Field(..., description="Similarity score of the match")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata of the document")

class SearchRequest(BaseModel):
    """Schema for a search request."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=5, description="Maximum number of results to return")

class SearchResponse(BaseModel):
    """Schema for search results."""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results found")

class QuestionResponse(BaseModel):
    """Schema for question answering response."""
    answer: str = Field(..., description="The generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used to generate the answer")

class SimplifyRequest(BaseModel):
    """Schema for text simplification request."""
    text: str = Field(..., description="The text to be simplified")
    context_id: Optional[str] = Field(None, description="Context ID for conversation grouping") 