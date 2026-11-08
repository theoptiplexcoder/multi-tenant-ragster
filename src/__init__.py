from fastapi import FastAPI
from src.embeddings.routes import embeddings_router

version = "v1"

description = """
A RAG application by theoptiplexcoder
    """

version_prefix =f"/api/{version}"

app = FastAPI(
    title="Ragster",
    description=description,
    version=version,
)

app.include_router(embeddings_router, prefix=f"{version_prefix}/embeddings", tags=["embeddings"])

@app.get("/health")
async def root():
    return {"status": "ok"}