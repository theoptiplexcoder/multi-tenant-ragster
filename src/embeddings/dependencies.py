from functools import lru_cache
from .service import AzureEmbeddingService

@lru_cache
def get_embedding_service() -> AzureEmbeddingService:
    """Provides a singleton instance of AzureEmbeddingService across requests."""
    return AzureEmbeddingService()