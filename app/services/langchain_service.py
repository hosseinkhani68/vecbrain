from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from app.config import get_settings
from typing import List, Dict, Any
import uuid

settings = get_settings()

# Initialize OpenAI components
llm = ChatOpenAI(
    model_name="gpt-4-1106-preview",
    temperature=0.7,
    max_tokens=1000
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.openai_api_key
)

# Custom prompt template for better conversation flow
CONVERSATION_TEMPLATE = """You are an AI assistant with a deep understanding of various topics.
You maintain context from previous conversations and provide detailed, accurate responses.

Current conversation:
{history}
Human: {input}
Assistant: Let me help you with that. """

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=CONVERSATION_TEMPLATE
)

class LangChainService:
    def __init__(self):
        self.conversations: Dict[str, ConversationChain] = {}
        self.vector_store = None
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize Qdrant vector store."""
        self.vector_store = Qdrant(
            client=settings.qdrant_client,
            collection_name="conversations",
            embeddings=embeddings
        )

    def get_or_create_conversation(self, context_id: str) -> ConversationChain:
        """Get existing conversation or create new one."""
        if context_id not in self.conversations:
            # Create retriever memory
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 5}  # Get 5 most relevant previous messages
            )
            memory = VectorStoreRetrieverMemory(
                retriever=retriever,
                memory_key="history",
                return_docs=True
            )

            # Create conversation chain
            conversation = ConversationChain(
                llm=llm,
                memory=memory,
                prompt=prompt,
                verbose=True
            )
            self.conversations[context_id] = conversation

        return self.conversations[context_id]

    async def process_message(self, text: str, context_id: str = None) -> Dict[str, Any]:
        """Process a message and return the response."""
        # Generate new context ID if not provided
        if not context_id:
            context_id = str(uuid.uuid4())

        # Get or create conversation
        conversation = self.get_or_create_conversation(context_id)

        # Process message
        response = conversation.predict(input=text)

        # Store in vector store
        await self._store_interaction(context_id, text, response)

        return {
            "id": context_id,
            "text": response,
            "role": "assistant",
            "timestamp": str(uuid.uuid4()),  # Using UUID as timestamp for simplicity
            "context_id": context_id
        }

    async def _store_interaction(self, context_id: str, user_message: str, assistant_response: str):
        """Store conversation interaction in vector store."""
        # Create documents for both messages
        user_doc = Document(
            page_content=user_message,
            metadata={
                "type": "chat",
                "chat_id": context_id,
                "role": "user",
                "timestamp": str(uuid.uuid4())
            }
        )

        assistant_doc = Document(
            page_content=assistant_response,
            metadata={
                "type": "chat",
                "chat_id": context_id,
                "role": "assistant",
                "timestamp": str(uuid.uuid4())
            }
        )

        # Add to vector store
        self.vector_store.add_documents([user_doc, assistant_doc])

    async def get_chat_history(self, context_id: str = None) -> List[Dict[str, Any]]:
        """Get chat history for a specific context or all contexts."""
        # Query vector store
        if context_id:
            # Get specific conversation
            results = self.vector_store.similarity_search(
                query="",  # Empty query to get all messages
                filter={"chat_id": context_id}
            )
        else:
            # Get all conversations
            results = self.vector_store.similarity_search(
                query="",  # Empty query to get all messages
                k=100  # Limit to 100 messages
            )

        # Convert to chat messages
        messages = []
        for doc in results:
            messages.append({
                "id": doc.metadata["chat_id"],
                "text": doc.page_content,
                "role": doc.metadata["role"],
                "timestamp": doc.metadata["timestamp"]
            })

        # Sort by timestamp
        messages.sort(key=lambda x: x["timestamp"])
        return messages 