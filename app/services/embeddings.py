from openai import AzureOpenAI

from app.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_EMBEDDINGS_API_VERSION,
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)

_client: AzureOpenAI | None = None

BATCH_SIZE = 20


def get_openai_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_EMBEDDINGS_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in batches, returning one embedding per text."""
    client = get_openai_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(
            input=batch,
            model=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_single(text: str) -> list[float]:
    """Embed a single text string."""
    return embed_texts([text])[0]
