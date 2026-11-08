from fastapi import APIRouter, Depends, status

from .dependencies import get_embedding_service
from .service import AzureEmbeddingService
from .schemas import (
    EmbeddingSingleRequest,
    EmbeddingBatchRequest,
    EmbeddingResponse,
)

embeddings_router = APIRouter()

@embeddings_router.post(
    "/single",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
)
async def create_single_embedding(
    payload: EmbeddingSingleRequest,
    service: AzureEmbeddingService = Depends(get_embedding_service),
) -> EmbeddingResponse:
    """Generate an embedding vector for a single piece of text."""
    return await service.generate_embeddings([payload.text])


@embeddings_router.post(
    "/batch",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
)
async def create_batch_embeddings(
    payload: EmbeddingBatchRequest,
    service: AzureEmbeddingService = Depends(get_embedding_service),
) -> EmbeddingResponse:
    """Generate embedding vectors for a batch of text strings."""
    return await service.generate_embeddings(payload.texts)