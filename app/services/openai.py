from openai import OpenAI
from app.config import get_settings
from typing import List, Dict
from functools import lru_cache
import tiktoken
from openai import OpenAIError

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

# Initialize tokenizer for text-embedding-3-small
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text."""
    return len(tokenizer.encode(text))

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> list[float]:
    """Get embedding for a text using OpenAI's API.
    
    Args:
        text: The text to embed
        
    Returns:
        list[float]: The embedding vector
        
    Raises:
        ValueError: If text is too long (exceeds 8191 tokens)
        OpenAIError: If there's an error with the OpenAI API
    """
    # Check token count
    token_count = count_tokens(text)
    if token_count > 8191:  # text-embedding-3-small limit
        raise ValueError(f"Text is too long. Maximum 8191 tokens allowed, got {token_count}")
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"  # Ensure we get float values
        )
        return response.data[0].embedding
    except OpenAIError as e:
        # Log the error and re-raise
        print(f"OpenAI API error: {str(e)}")
        raise
    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error: {str(e)}")
        raise

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
        model="gpt-4-1106-preview",  # GPT-4 Turbo
        messages=messages,
        temperature=0.7,  # Add some creativity while keeping responses focused
        max_tokens=1000,  # Limit response length to control costs
        top_p=0.95,  # Nucleus sampling for better response quality
        frequency_penalty=0.0,  # Don't penalize frequent tokens
        presence_penalty=0.0  # Don't penalize new tokens
    )
    return response.choices[0].message.content 