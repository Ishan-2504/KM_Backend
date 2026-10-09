from google import genai

from app.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

_client: genai.Client | None = None

BATCH_SIZE = 20


def get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts one at a time so Gemini does not collapse a batch into one vector."""
    client = get_gemini_client()
    all_embeddings: list[list[float]] = []

    for text in texts:
        response = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=text or " ",
        )
        if not response.embeddings:
            raise RuntimeError("Gemini embed_content returned no embeddings")
        all_embeddings.append(list(response.embeddings[0].values))

    return all_embeddings


def embed_single(text: str) -> list[float]:
    """Embed a single text string."""
    return embed_texts([text])[0]
