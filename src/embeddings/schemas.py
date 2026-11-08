from typing import List, Optional
from pydantic import BaseModel, Field

class EmbeddingSingleRequest(BaseModel):
    """Schema for embedding a single text string."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="The text content to generate an embedding for.",
        examples=["FastAPI is a modern, fast web framework for building APIs."]
    )

class EmbeddingBatchRequest(BaseModel):
    """Schema for embedding a batch of text strings."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="List of text strings to generate embeddings for.",
    )

class EmbeddingItem(BaseModel):
    """Schema representing an individual embedding output."""
    index: int = Field(description="The index position matching the input array.")
    embedding: List[float] = Field(description="The dense vector embedding.")

class EmbeddingUsage(BaseModel):
    """Schema representing token usage metadata from the provider."""
    prompt_tokens: int = Field(default=0, description="Number of tokens processed.")
    total_tokens: int = Field(default=0, description="Total tokens consumed.")

class EmbeddingResponse(BaseModel):
    """Response schema for embedding operations."""
    data: List[EmbeddingItem] = Field(description="List of embedding vectors.")
    model: str = Field(description="The embedding deployment/model used.")
    dimension: int = Field(description="Dimensionality of the embedding vectors (e.g., 1536).")
    usage: Optional[EmbeddingUsage] = Field(default=None, description="Token consumption metrics.")