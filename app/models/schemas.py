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

class DocumentProcessRequest(BaseModel):
    """Request model for document processing."""
    file_path: str = Field(..., description="Path to the document file")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the document")

class DocumentProcessResponse(BaseModel):
    """Response model for document processing."""
    doc_id: str = Field(..., description="Unique identifier for the processed document")
    chunks: int = Field(..., description="Number of chunks the document was split into")
    source: str = Field(..., description="Source file path")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")

class DocumentSearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=5, description="Maximum number of results to return")

class DocumentSearchResponse(BaseModel):
    """Response model for document search."""
    results: List[Dict[str, Any]] = Field(..., description="Search results with text and metadata")

class DocumentChunksRequest(BaseModel):
    """Request model for retrieving document chunks."""
    doc_id: str = Field(..., description="Document ID to retrieve chunks for")

class DocumentChunksResponse(BaseModel):
    """Response model for document chunks."""
    chunks: List[Dict[str, Any]] = Field(..., description="Document chunks with text and metadata")

class AgentQueryRequest(BaseModel):
    """Request model for agent queries."""
    query: str = Field(..., description="The query to process using the agent")

class AgentQueryResponse(BaseModel):
    """Response model for agent queries."""
    response: str = Field(..., description="The agent's response")
    tools_used: List[str] = Field(default=[], description="List of tools used by the agent")
    timestamp: str = Field(..., description="Timestamp of the response")
    error: Optional[str] = Field(None, description="Error message if any")

class PromptRequest(BaseModel):
    """Request model for prompt generation."""
    template_name: str = Field(..., description="Name of the template to use")
    input_data: Dict[str, Any] = Field(..., description="Input data for the template")
    context: Optional[str] = Field(None, description="Additional context for the prompt")
    history: Optional[List[Dict[str, str]]] = Field(None, description="Conversation history")

class PromptResponse(BaseModel):
    """Response model for prompt generation."""
    response: str = Field(..., description="Generated response")
    template_used: str = Field(..., description="Name of the template used")
    timestamp: str = Field(..., description="Timestamp of the response")
    error: Optional[str] = Field(None, description="Error message if any")

class TemplateInfoResponse(BaseModel):
    """Response model for template information."""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    input_variables: List[str] = Field(..., description="Required input variables")

class StreamResponse(BaseModel):
    """Model for streaming response chunks."""
    chunk: str = Field(..., description="A chunk of the streaming response")
    done: bool = Field(default=False, description="Whether this is the final chunk")
    error: Optional[str] = Field(None, description="Error message if any") 