from openai import OpenAI
from app.config import get_settings

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

def get_embedding(text: str) -> list[float]:
    """Get embedding for a text using OpenAI's API."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def get_completion(prompt: str) -> str:
    """Get completion from OpenAI's API."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content 