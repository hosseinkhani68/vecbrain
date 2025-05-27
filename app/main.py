from fastapi import FastAPI, HTTPException, UploadFile, File, Request
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
import asyncio
import logging
import time

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
    expose_headers=["*"]  # Expose all headers
)

# Initialize LangChain service
langchain_service = LangChainService()

# Initialize document service
document_service = DocumentService()

# Initialize agent service
agent_service = AgentService(document_service)

# Initialize prompt service
prompt_service = PromptService()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request details
    logger.info(f"\n{'='*50}")
    logger.info(f"Request started: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    
    # Try to log request body for POST requests
    if request.method == "POST":
        try:
            body = await request.body()
            if body:
                try:
                    body_json = json.loads(body)
                    logger.info(f"Request body: {json.dumps(body_json, indent=2)}")
                except:
                    logger.info(f"Request body: {body.decode()}")
        except:
            logger.info("Could not read request body")
    
    # Process the request
    response = await call_next(request)
    
    # Log response details
    process_time = time.time() - start_time
    logger.info(f"Request completed in {process_time:.2f}s")
    logger.info(f"Response status: {response.status_code}")
    logger.info(f"{'='*50}\n")
    
    return response

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
    description="""
    Retrieve the chat history with pagination support.
    
    Parameters:
    - context_id: Optional conversation ID to filter messages
    - limit: Number of messages to return (default: 10)
    - offset: Number of messages to skip (default: 0)
    """
)
async def get_chat_history(
    context_id: str = None,
    limit: int = 10,
    offset: int = 0
):
    """Get the chat history with pagination."""
    try:
        print(f"Received chat history request - context_id: {context_id}, limit: {limit}, offset: {offset}")
        
        # Add timeout handling with reduced timeout
        messages = await asyncio.wait_for(
            langchain_service.get_chat_history(
                context_id=context_id,
                limit=limit,
                offset=offset
            ),
            timeout=5.0  # Reduced timeout to 5 seconds
        )
        
        if not messages:
            print("No messages found, returning empty response")
            return [ChatHistoryResponse(messages=[])]
            
        print(f"Returning {len(messages)} messages")
        return [ChatHistoryResponse(messages=messages)]
    except asyncio.TimeoutError:
        print("Request timed out while fetching chat history")
        return [ChatHistoryResponse(messages=[])]
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return [ChatHistoryResponse(messages=[])]

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with streaming support."""
    try:
        # Log detailed request information
        logger.info("\n=== Incoming Chat Request ===")
        logger.info(f"Request ID: {str(uuid.uuid4())}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Context ID: {request.context_id}")
        logger.info(f"Stream Mode: {request.stream}")
        logger.info(f"Text Length: {len(request.text) if request.text else 0}")
        logger.info(f"Text Preview: {request.text[:100] + '...' if request.text and len(request.text) > 100 else request.text}")
        
        # Initialize conversation history if needed
        if not langchain_service.chat_history:
            langchain_service.chat_history = []
        
        # Process text input
        if request.text:
            # Split long text into chunks if needed
            chunks = []
            current_chunk = ""
            for word in request.text.split():
                if len(current_chunk) + len(word) + 1 <= 500:  # 500 char limit per chunk
                    current_chunk += " " + word if current_chunk else word
                else:
                    chunks.append(current_chunk)
                    current_chunk = word
            if current_chunk:
                chunks.append(current_chunk)
            
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Process each chunk
            responses = []
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    # Set a timeout for each chunk
                    async with asyncio.timeout(15.0):  # 15 second timeout per chunk
                        response = await langchain_service.get_chat_response(chunk)
                        responses.append(response)
                        logger.info(f"Chunk {i+1} processed successfully")
                except asyncio.TimeoutError:
                    logger.error(f"Timeout processing chunk {i+1}: {chunk[:50]}...")
                    responses.append("I apologize, but this part of your message took too long to process. Please try again with a shorter message.")
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    responses.append("I apologize, but I encountered an error processing this part of your message.")
            
            # Combine responses
            final_response = " ".join(responses)
            logger.info(f"Final response length: {len(final_response)}")
            
            # Return streaming response if requested
            if request.stream:
                logger.info("Returning streaming response")
                async def generate():
                    for chunk in final_response.split():
                        yield f"data: {chunk}\n\n"
                        await asyncio.sleep(0.1)  # Small delay between chunks
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream"
                )
            
            logger.info("Returning regular response")
            return {"response": final_response}
        
        logger.info("No text provided in request")
        return {"response": "No text provided"}
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

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