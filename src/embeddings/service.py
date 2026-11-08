import logging
from typing import List, Sequence
from openai import AsyncAzureOpenAI, APIError

from src.config import settings
from src.errors import EmbeddingServiceError
from .schemas import EmbeddingItem, EmbeddingResponse, EmbeddingUsage

logger = logging.getLogger(__name__)


class AzureEmbeddingService:
    def __init__(self):
        self.deployment_name = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        self.client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

    def _sanitize_text(self, text: str) -> str:
        """Sanitizes text per OpenAI guidance (replace newlines with space)."""
        return text.replace("\n", " ").strip()

    async def generate_embeddings(self, texts: Sequence[str]) -> EmbeddingResponse:
        """
        Calls Azure OpenAI asynchronously to create vector embeddings.
        Returns a structured EmbeddingResponse.
        """
        if not texts:
            return EmbeddingResponse(
                data=[],
                model=self.deployment_name,
                dimension=0,
                usage=EmbeddingUsage()
            )

        sanitized_inputs = [self._sanitize_text(t) for t in texts]

        try:
            response = await self.client.embeddings.create(
                input=sanitized_inputs,
                model=self.deployment_name,
            )

            # Sort items by their reported index to guarantee input alignment
            sorted_data = sorted(response.data, key=lambda item: item.index)

            items = [
                EmbeddingItem(index=item.index, embedding=item.embedding)
                for item in sorted_data
            ]

            dimension = len(items[0].embedding) if items else 0

            usage = None
            if response.usage:
                usage = EmbeddingUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            return EmbeddingResponse(
                data=items,
                model=response.model or self.deployment_name,
                dimension=dimension,
                usage=usage,
            )

        except APIError as e:
            logger.error(f"Azure OpenAI API error occurred: {e}", exc_info=True)
            raise EmbeddingServiceError()
        except Exception as e:
            logger.error(f"Unexpected error in embedding generation: {e}", exc_info=True)
            raise EmbeddingServiceError()