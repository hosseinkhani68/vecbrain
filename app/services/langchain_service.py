from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from app.config import get_settings
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import uuid

settings = get_settings()

class LangChainService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
        self.vector_store = Qdrant(
            client=settings.qdrant_client,
            collection_name="documents",
            embeddings=self.embeddings
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
        self.chat_history: List[Dict[str, str]] = []

    async def get_chat_history(self, context_id: Optional[str] = None) -> List[Dict[str, str]]:
        """Get the chat history from Qdrant.
        
        Args:
            context_id: Optional context ID to filter messages by conversation.
            
        Returns:
            List of chat messages, each containing id, text, role, and timestamp.
        """
        try:
            # Create filter for chat messages
            filter_conditions = {
                "type": "chat"
            }
            if context_id:
                filter_conditions["context_id"] = context_id

            # Search for chat messages in Qdrant
            results = self.vector_store.similarity_search(
                query="",  # Empty query to get all messages
                k=100,  # Get last 100 messages
                filter=filter_conditions
            )
            
            # Convert results to chat messages
            messages = []
            for doc in results:
                if "type" in doc.metadata and doc.metadata["type"] == "chat":
                    messages.append({
                        "id": doc.metadata.get("message_id", str(uuid.uuid4())),
                        "text": doc.page_content,
                        "role": doc.metadata.get("role", "user"),
                        "timestamp": doc.metadata.get("timestamp", datetime.now().isoformat()),
                        "context_id": doc.metadata.get("context_id")
                    })
            
            # Sort by timestamp in ascending order
            messages.sort(key=lambda x: x["timestamp"])
            return messages
        except Exception as e:
            print(f"Error getting chat history: {str(e)}")
            return []

    async def add_to_chat_history(self, role: str, content: str, context_id: Optional[str] = None) -> None:
        """Add a message to the chat history in Qdrant."""
        try:
            message_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Ensure context_id is set
            if not context_id:
                context_id = str(uuid.uuid4())
            
            # Store in Qdrant with all required metadata
            self.vector_store.add_texts(
                texts=[content],
                metadatas=[{
                    "type": "chat",
                    "message_id": message_id,
                    "role": role,
                    "timestamp": timestamp,
                    "context_id": context_id,
                    "content": content  # Store content in metadata for easier retrieval
                }]
            )
            
            # Also add to in-memory history for current session
            self.chat_history.append({
                "id": message_id,
                "text": content,
                "role": role,
                "timestamp": timestamp,
                "context_id": context_id
            })
        except Exception as e:
            print(f"Error adding to chat history: {str(e)}")
            raise e  # Re-raise the exception to handle it in the API endpoint

    async def clear_chat_history(self, context_id: Optional[str] = None) -> None:
        """Clear the chat history from Qdrant."""
        try:
            # Delete messages from Qdrant
            self.vector_store.delete(
                filter={
                    "type": "chat",
                    **({"context_id": context_id} if context_id else {})
                }
            )
        except Exception as e:
            print(f"Error clearing chat history: {str(e)}")

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

    async def get_chat_response(self, query: str) -> str:
        """Get a response from the chat model with document context."""
        try:
            # Create conversation chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(),
                return_source_documents=True,
                verbose=True  # Add verbose mode for debugging
            )

            # Format chat history
            formatted_history = []
            for msg in self.chat_history:
                if msg["role"] == "user":
                    formatted_history.append((msg["text"], ""))
                elif msg["role"] == "assistant":
                    if formatted_history:
                        formatted_history[-1] = (formatted_history[-1][0], msg["text"])

            # Get response with error handling
            try:
                result = chain({"question": query, "chat_history": formatted_history})
                if not result or "answer" not in result:
                    # If no answer is generated, use the LLM directly
                    response = await self.llm.ainvoke([{"role": "user", "content": query}])
                    answer = response.content
                else:
                    answer = result["answer"]
            except Exception as chain_error:
                # Fallback to direct LLM response if chain fails
                response = await self.llm.ainvoke([{"role": "user", "content": query}])
                answer = response.content
            
            # Add to chat history
            await self.add_to_chat_history("user", query)
            await self.add_to_chat_history("assistant", answer)
            
            return answer
        except Exception as e:
            raise Exception(f"Error getting chat response: {str(e)}") 