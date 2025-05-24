from openai import OpenAI
from app.config import get_settings
from typing import List, Dict

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

def get_embedding(text: str) -> list[float]:
    """Get embedding for a text using OpenAI's API."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def get_completion(prompt: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """Get completion from OpenAI's API with conversation history support.
    
    Args:
        prompt: The current user message
        conversation_history: List of previous messages in the format [{"role": "user/assistant", "content": "message"}]
    """
    messages = []
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add the current prompt
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content 