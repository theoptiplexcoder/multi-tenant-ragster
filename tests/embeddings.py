from unittest.mock import AsyncMock
import pytest
from src.embeddings.dependencies import get_embedding_service
from src import app
from src.embeddings.schemas import EmbeddingResponse, EmbeddingItem, EmbeddingUsage
from src.config import settings

embeddings_prefix = "/api/v1/embeddings"

@pytest.fixture
def mock_embedding_service():
    service = AsyncMock()
    # Mock a deterministic response
    service.generate_embeddings.return_value = EmbeddingResponse(
        data=[
            EmbeddingItem(index=0, embedding=[0.1, 0.2, 0.3]),
            EmbeddingItem(index=1, embedding=[0.4, 0.5, 0.6]),
        ],
        model="text-embedding-3-small",
        dimension=3,
        usage=EmbeddingUsage(prompt_tokens=10, total_tokens=10),
    )
    return service


def test_create_single_embedding(test_client, mock_embedding_service):
    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service

    payload = {"text": "FastAPI with Azure OpenAI"}
    response = test_client.post(f"{embeddings_prefix}/single", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "text-embedding-3-small"
    assert data["dimension"] == 3
    assert len(data["data"]) == 2

    app.dependency_overrides.pop(get_embedding_service, None)


def test_create_batch_embeddings(test_client, mock_embedding_service):
    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service

    payload = {"texts": ["Text one", "Text two"]}
    response = test_client.post(f"{embeddings_prefix}/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 2
    assert data["data"][0]["index"] == 0

    app.dependency_overrides.pop(get_embedding_service, None)


def test_embedding_validation_empty_text(test_client):
    # Tests Pydantic min_length validation (should reject with 422 Unprocessable Entity)
    response = test_client.post(f"{embeddings_prefix}/single", json={"text": ""})
    assert response.status_code == 422