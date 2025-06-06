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
import asyncio

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

    async def get_chat_history(
        self,
        context_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, str]]:
        """Get the chat history from Qdrant with pagination."""
        try:
            print(f"Fetching chat history - context_id: {context_id}, limit: {limit}, offset: {offset}")
            
            # Create filter for chat messages
            filter_conditions = {
                "type": "chat"
            }
            if context_id:
                filter_conditions["context_id"] = context_id

            # Use Qdrant's search method with pagination and optimized parameters
            start_time = datetime.now()
            results = await self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                filter=filter_conditions,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                timeout=3.0  # Reduced timeout for faster response
            )
            end_time = datetime.now()
            print(f"Qdrant query took {(end_time - start_time).total_seconds()} seconds")

            if not results or not results[0]:
                print("No results found")
                return []

            # Convert results to chat messages more efficiently
            messages = []
            for point in results[0]:
                payload = point.payload
                if not payload or "type" not in payload or payload["type"] != "chat":
                    continue
                    
                messages.append({
                    "id": payload.get("message_id", str(uuid.uuid4())),
                    "text": payload.get("content", ""),
                    "role": payload.get("role", "user"),
                    "timestamp": payload.get("timestamp", datetime.now().isoformat()),
                    "context_id": payload.get("context_id")
                })
            
            # Sort by timestamp in ascending order
            messages.sort(key=lambda x: x["timestamp"])
            print(f"Returning {len(messages)} messages")
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
                texts=[content],  # Store content in the text field
                metadatas=[{
                    "type": "chat",
                    "message_id": message_id,
                    "role": role,
                    "timestamp": timestamp,
                    "context_id": context_id,
                    "content": content  # Also store content in metadata for easier retrieval
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
            print(f"Processing chat query: {query[:50]}...")  # Log first 50 chars
            
            # Set a timeout for the entire operation
            async with asyncio.timeout(10.0):  # 10 second timeout
                # Create conversation chain with optimized parameters
                chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vector_store.as_retriever(
                        search_kwargs={"k": 3}  # Limit to 3 most relevant documents
                    ),
                    return_source_documents=True,
                    verbose=False  # Disable verbose mode for better performance
                )

                # Format chat history (limit to last 5 messages for context)
                formatted_history = []
                recent_history = self.chat_history[-5:] if self.chat_history else []
                for msg in recent_history:
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
                    print(f"Chain error, falling back to direct LLM: {str(chain_error)}")
                    # Fallback to direct LLM response if chain fails
                    response = await self.llm.ainvoke([{"role": "user", "content": query}])
                    answer = response.content
                
                # Add to chat history asynchronously
                asyncio.create_task(self.add_to_chat_history("user", query))
                asyncio.create_task(self.add_to_chat_history("assistant", answer))
                
                return answer
        except asyncio.TimeoutError:
            print("Chat response timed out")
            return "I apologize, but the request took too long to process. Please try again with a shorter message or different query."
        except Exception as e:
            print(f"Error getting chat response: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Please try again." 