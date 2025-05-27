from openai import OpenAI
from app.config import get_settings
from typing import List, Dict, AsyncGenerator
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

async def get_completion_stream(prompt: str, conversation_history: List[tuple] = None) -> AsyncGenerator[str, None]:
    """Get streaming completion from OpenAI's API with conversation history support.
    
    Args:
        prompt: The current user message
        conversation_history: List of previous messages in the format [(user_msg, assistant_msg)]
        
    Yields:
        str: Chunks of the response as they are generated
    """
    messages = []
    
    # Add conversation history if provided
    if conversation_history:
        for user_msg, assistant_msg in conversation_history:
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add the current prompt
    messages.append({"role": "user", "content": prompt})
    
    try:
        stream = client.chat.completions.create(
            model="gpt-4-1106-preview",  # GPT-4 Turbo
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=True  # Enable streaming
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        print(f"Error in streaming completion: {str(e)}")
        yield f"Error: {str(e)}"

# Update the existing get_completion to use the new streaming function
async def get_completion(prompt: str, conversation_history: List[Dict[str, str]] = None) -> str:
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
    
    response = await client.chat.completions.create(
        model="gpt-4-1106-preview",  # GPT-4 Turbo
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
        top_p=0.95,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].message.content 