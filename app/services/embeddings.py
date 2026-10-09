from google import genai

from app.config import (
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
)

_client: genai.Client | None = None

BATCH_SIZE = 20


def get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=GEMINI_API_KEY
        )
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in batches, returning one embedding per text."""
    client = get_gemini_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.models.embed(
            model=GEMINI_EMBEDDING_MODEL,
            contents=batch,
        )
        batch_embeddings = [item.values for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_single(text: str) -> list[float]:
    """Embed a single text string."""
    return embed_texts([text])[0]
