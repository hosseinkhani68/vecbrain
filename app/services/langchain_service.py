from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from app.config import get_settings
from typing import List, Dict, Any
import os

settings = get_settings()

class LangChainService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
        self.vector_store = QdrantVectorStore(
            collection_name="documents",
            embedding_function=self.embeddings.embed_query,
            client=settings.qdrant_client
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.llm = ChatOpenAI(
            model="gpt-4-1106-preview",
            temperature=0.7,
            openai_api_key=settings.openai_api_key
        )

    async def process_document(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a document and store it in the vector store."""
        try:
            # Load and split document
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = self.text_splitter.split_text(text)

            # Add metadata
            doc_id = os.path.basename(file_path)
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_id": i,
                        "source": file_path,
                        **(metadata or {})
                    }
                })

            # Store in vector store
            self.vector_store.add_texts(
                texts=[doc["text"] for doc in documents],
                metadatas=[doc["metadata"] for doc in documents]
            )

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

    async def get_chat_response(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Get a response from the chat model with document context."""
        try:
            # Create conversation chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(),
                return_source_documents=True
            )

            # Format chat history
            formatted_history = []
            if chat_history:
                for msg in chat_history:
                    formatted_history.append((msg["user"], msg["assistant"]))

            # Get response
            result = chain({"question": query, "chat_history": formatted_history})
            return result["answer"]
        except Exception as e:
            raise Exception(f"Error getting chat response: {str(e)}") 