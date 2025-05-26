from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.services import openai, qdrant
from app.models.schemas import (
    DocumentCreate,
    DocumentResponse,
    SearchQuery,
    SearchResponse,
    QuestionResponse,
    SimplifyRequest,
    ChatMessage,
    ChatResponse,
    ChatRequest,
    ChatHistoryResponse,
    DocumentProcessRequest,
    DocumentProcessResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentChunksRequest,
    DocumentChunksResponse,
    AgentQueryRequest,
    AgentQueryResponse,
    PromptRequest,
    PromptResponse,
    TemplateInfoResponse
)
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from app.services.langchain_service import LangChainService
from app.services.document_service import DocumentService
from app.services.agent_service import AgentService
from app.services.prompt_service import PromptService
import os
import shutil
import json
from app.services.openai import get_completion_stream

app = FastAPI(
    title="VecBrain API",
    description="""
    A powerful API for semantic search and question answering using OpenAI and Qdrant.
    
    ## Features
    * Store documents with embeddings
    * Search for similar documents
    * Ask questions and get AI-powered answers
    * Simplify complex text
    
    ## Authentication
    Currently, this API is open. In production, you should add authentication.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize LangChain service
langchain_service = LangChainService()

# Initialize document service
document_service = DocumentService()

# Initialize agent service
agent_service = AgentService(document_service)

# Initialize prompt service
prompt_service = PromptService()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic API information."""
    return """
    <html>
        <head>
            <title>VecBrain API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }
                h1 { color: #2c3e50; }
                .endpoint {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .method { color: #e74c3c; font-weight: bold; }
                .url { color: #3498db; }
            </style>
        </head>
        <body>
            <h1>Welcome to VecBrain API</h1>
            <p>A powerful API for semantic search and question answering using OpenAI and Qdrant.</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="url">/documents</span>
                <p>Store a new document with its embedding.</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="url">/search</span>
                <p>Search for similar documents.</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="url">/ask</span>
                <p>Ask a question and get an AI-powered answer.</p>
            </div>
            
            <h2>Documentation:</h2>
            <ul>
                <li><a href="/docs">Swagger UI Documentation</a></li>
                <li><a href="/redoc">ReDoc Documentation</a></li>
            </ul>
        </body>
    </html>
    """

@app.post(
    "/documents",
    response_model=DocumentResponse,
    summary="Store a new document",
    description="""
    Store a new document in the vector database.
    
    The document will be:
    1. Converted to embeddings using OpenAI
    2. Stored in Qdrant with its metadata
    3. Returned with its ID and embedding
    """,
    response_description="The stored document with its ID and embedding"
)
async def store_document(document: DocumentCreate):
    """Store a document with its embedding."""
    try:
        embedding = openai.get_embedding(document.text)
        doc_id = str(uuid.uuid4())  # Generate a UUID for the document
        await qdrant.store_document(doc_id, document.text, embedding, document.metadata)
        return DocumentResponse(
            id=doc_id,
            text=document.text,
            metadata=document.metadata,
            embedding=embedding
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Search for similar documents",
    description="""
    Search for documents similar to the query text.
    
    The search:
    1. Converts the query to embeddings
    2. Finds similar documents in Qdrant
    3. Returns results with similarity scores
    """,
    response_description="List of similar documents with their scores"
)
async def search_documents(query: SearchQuery):
    """Search for similar documents."""
    try:
        query_embedding = openai.get_embedding(query.text)
        results = await qdrant.search_similar(query_embedding, query.limit)
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/ask",
    response_model=QuestionResponse,
    summary="Ask a question",
    description="""
    Ask a question and get an AI-powered answer based on similar documents.
    
    The process:
    1. Finds similar documents to the question
    2. Uses them as context
    3. Generates an answer using OpenAI
    4. Returns the answer with source documents
    """,
    response_description="The generated answer with source documents"
)
async def ask_question(query: SearchQuery):
    """Ask a question and get an answer based on similar documents."""
    try:
        # Get similar documents
        query_embedding = openai.get_embedding(query.text)
        similar_docs = await qdrant.search_similar(query_embedding, query.limit)
        
        # Create context from similar documents
        context = "\n".join(doc["text"] for doc in similar_docs)
        
        # Generate answer using OpenAI
        prompt = f"""Based on the following context, please answer the question.
        
Context:
{context}

Question: {query.text}

Answer:"""
        
        answer = openai.get_completion(prompt)
        return QuestionResponse(
            answer=answer,
            sources=similar_docs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/simplify",
    response_model=QuestionResponse,
    summary="Simplify text",
    description="""
    Simplify complex text to make it easier to understand.
    
    The process:
    1. Takes the input text
    2. Uses previous chat history as context
    3. Uses OpenAI to generate a simplified version
    4. Stores the interaction in chat history
    5. Returns the simplified text
    """,
    response_description="The simplified text"
)
async def simplify_text(request: SimplifyRequest):
    """Simplify text to make it easier to understand."""
    try:
        # Get recent chat history for context
        query_embedding = openai.get_embedding(request.text)
        recent_chats = await qdrant.search_similar(
            query_embedding=query_embedding,
            limit=5  # Get 5 most relevant previous simplifications
        )
        
        # Create context from recent chats
        context = ""
        if recent_chats:
            context = "Previous simplifications for context:\n"
            for chat in recent_chats:
                if "type" in chat["metadata"] and chat["metadata"]["type"] == "chat":
                    context += f"\nOriginal: {chat['text']}\n"
                    # Find the corresponding simplified version
                    simplified_chats = await qdrant.search_similar(
                        query_embedding=openai.get_embedding(chat["text"]),
                        limit=1
                    )
                    if simplified_chats:
                        context += f"Simplified: {simplified_chats[0]['text']}\n"
        
        prompt = f"""Please simplify the following text to make it easier to understand. 
        Keep the main ideas but use simpler language and shorter sentences.
        Use the previous simplifications as context to maintain consistency in simplification style.

{context}

Text to simplify:
{request.text}

Simplified text:"""
        
        simplified = openai.get_completion(prompt)
        
        # Store in chat history
        chat_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Store original text
        original_embedding = openai.get_embedding(request.text)
        await qdrant.store_document(
            doc_id=str(uuid.uuid4()),
            text=request.text,
            embedding=original_embedding,
            metadata={
                "type": "chat",
                "chat_id": chat_id,
                "role": "user",
                "timestamp": timestamp
            }
        )
        
        # Store simplified text
        simplified_embedding = openai.get_embedding(simplified)
        await qdrant.store_document(
            doc_id=str(uuid.uuid4()),
            text=simplified,
            embedding=simplified_embedding,
            metadata={
                "type": "chat",
                "chat_id": chat_id,
                "role": "assistant",
                "timestamp": timestamp
            }
        )
        
        return QuestionResponse(
            answer=simplified,
            sources=recent_chats  # Include relevant previous simplifications as sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/chat-history",
    response_model=List[ChatHistoryResponse],
    summary="Get chat history",
    description="Retrieve the chat history with simplified texts."
)
async def get_chat_history(context_id: str = None):
    """Get the chat history."""
    try:
        messages = await langchain_service.get_chat_history(context_id)
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Interactive chat endpoint",
    description="""
    Interactive chat endpoint that maintains conversation context and memory.
    The system will remember previous messages and use them to provide more contextual responses.
    """,
    response_description="The chat response with context"
)
async def chat(request: ChatRequest):
    """Chat with the AI assistant."""
    try:
        if request.stream:
            return StreamingResponse(
                generate(request.text, request.history),
                media_type="text/event-stream"
            )
        
        response = await langchain_service.get_chat_response(request.text)
        return ChatResponse(
            text=response,
            history=request.history + [
                ChatMessage(role="user", content=request.text),
                ChatMessage(role="assistant", content=response)
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate(text: str, history: List[ChatMessage]):
    """Generate streaming response."""
    try:
        async for chunk in get_completion_stream(text, history):
            yield f"data: {chunk}\n\n"
    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"

@app.post(
    "/documents/process",
    response_model=DocumentProcessResponse,
    summary="Process a document",
    description="Upload and process a document for semantic search."
)
async def process_document(
    file: UploadFile = File(...),
    metadata: Optional[Dict[str, Any]] = None
):
    """Process a document and store it in the vector store."""
    try:
        # Create temporary file
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        result = await document_service.process_document(temp_file, metadata)
        
        # Clean up
        os.remove(temp_file)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/documents/search",
    response_model=DocumentSearchResponse,
    summary="Search documents",
    description="Search for relevant document chunks using semantic search."
)
async def search_documents(request: DocumentSearchRequest):
    """Search for relevant document chunks."""
    try:
        results = await document_service.search_documents(
            query=request.query,
            limit=request.limit
        )
        return DocumentSearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/documents/chunks",
    response_model=DocumentChunksResponse,
    summary="Get document chunks",
    description="Retrieve all chunks for a specific document."
)
async def get_document_chunks(request: DocumentChunksRequest):
    """Get all chunks for a specific document."""
    try:
        chunks = await document_service.get_document_chunks(request.doc_id)
        return DocumentChunksResponse(chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/agent/query",
    response_model=AgentQueryResponse,
    summary="Query the agent",
    description="Send a query to the agent that can use various tools to answer."
)
async def query_agent(request: AgentQueryRequest):
    """Process a query using the agent and its tools."""
    try:
        result = await agent_service.process_complex_query(request.query)
        return AgentQueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/prompt/generate",
    response_model=PromptResponse,
    summary="Generate response using a prompt template",
    description="Generate a response using one of the available prompt templates."
)
async def generate_prompt_response(request: PromptRequest):
    """Generate a response using the specified prompt template."""
    try:
        result = await prompt_service.generate_response(
            template_name=request.template_name,
            input_data=request.input_data,
            context=request.context,
            history=request.history
        )
        return PromptResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/prompt/templates",
    response_model=List[str],
    summary="Get available templates",
    description="Get a list of available prompt template names."
)
async def get_templates():
    """Get a list of available prompt templates."""
    try:
        return prompt_service.get_available_templates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/prompt/templates/{template_name}",
    response_model=TemplateInfoResponse,
    summary="Get template information",
    description="Get detailed information about a specific prompt template."
)
async def get_template_info(template_name: str):
    """Get information about a specific prompt template."""
    try:
        info = prompt_service.get_template_info(template_name)
        return TemplateInfoResponse(**info)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 