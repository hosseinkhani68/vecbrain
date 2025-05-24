from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from app.services import openai, qdrant
from app.models.schemas import (
    DocumentCreate,
    DocumentResponse,
    SearchQuery,
    SearchResponse,
    QuestionResponse,
    SimplifyRequest
)

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
        doc_id = hash(document.text)
        await qdrant.store_document(document.text, embedding, document.metadata)
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
    2. Uses OpenAI to generate a simplified version
    3. Returns the simplified text
    """,
    response_description="The simplified text"
)
async def simplify_text(request: SimplifyRequest):
    """Simplify text to make it easier to understand."""
    try:
        prompt = f"""Please simplify the following text to make it easier to understand. 
        Keep the main ideas but use simpler language and shorter sentences:

Text to simplify:
{request.text}

Simplified text:"""
        
        simplified = openai.get_completion(prompt)
        return QuestionResponse(
            answer=simplified,
            sources=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 