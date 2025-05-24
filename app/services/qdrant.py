from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import get_settings

settings = get_settings()
client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key
)

COLLECTION_NAME = "documents"

async def init_collection():
    """Initialize the Qdrant collection if it doesn't exist."""
    collections = client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI text-embedding-3-small dimension
                distance=models.Distance.COSINE
            )
        )

async def store_document(text: str, embedding: list[float], metadata: dict = None):
    """Store a document with its embedding in Qdrant."""
    await init_collection()
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=hash(text),  # Simple hash as ID
                vector=embedding,
                payload={"text": text, **(metadata or {})}
            )
        ]
    )

async def search_similar(query_embedding: list[float], limit: int = 5):
    """Search for similar documents using the query embedding."""
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=limit
    )
    return [
        {
            "text": hit.payload["text"],
            "score": hit.score,
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
        }
        for hit in results
    ] 