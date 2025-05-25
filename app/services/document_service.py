from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from app.config import get_settings
from app.models.schemas import Document, DocumentChunk
from typing import List, Dict, Any, Optional
import os
import uuid
from datetime import datetime

settings = get_settings()

class DocumentService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name="documents",
            embeddings=self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def _get_loader(self, file_path: str):
        """Get appropriate loader based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        loaders = {
            '.txt': TextLoader,
            '.pdf': PDFMinerLoader,
            '.docx': Docx2txtLoader,
            '.csv': CSVLoader,
            '.md': UnstructuredMarkdownLoader,
            '.html': UnstructuredHTMLLoader,
        }
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: {ext}")
        return loaders[ext](file_path)

    async def process_document(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a document and store it in the vector store."""
        try:
            # Load document
            loader = self._get_loader(file_path)
            documents = loader.load()

            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            # Add metadata
            doc_id = str(uuid.uuid4())
            for chunk in chunks:
                chunk.metadata.update({
                    "doc_id": doc_id,
                    "source": file_path,
                    **(metadata or {})
                })

            # Store in vector store
            self.vector_store.add_documents(chunks)

            return {
                "doc_id": doc_id,
                "chunks": len(chunks),
                "source": file_path,
                "metadata": metadata
            }
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    async def search_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=limit
            )
            return [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", 0)
                }
                for doc in results
            ]
        except Exception as e:
            raise Exception(f"Error searching documents: {str(e)}")

    async def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        try:
            results = self.vector_store.similarity_search(
                query="",  # Empty query to get all chunks
                filter={"doc_id": doc_id}
            )
            return [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        except Exception as e:
            raise Exception(f"Error retrieving document chunks: {str(e)}") 