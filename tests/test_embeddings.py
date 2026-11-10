"""The OpenAI embedding client is pinned to 384 dimensions."""
from core.embeddings import EMBEDDING_DIM, _openai_client, embed_texts


def test_openai_embeddings_use_384_dimensions(settings, monkeypatch):
    settings.EMBEDDINGS_PROVIDER = "openai"
    settings.EMBEDDINGS_MODEL = ""
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def embed_documents(self, texts):
            return [[1.0] * EMBEDDING_DIM for _ in texts]

    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", FakeEmbeddings)
    _openai_client.cache_clear()
    try:
        vectors = embed_texts(["part 65-45210-3"])
    finally:
        _openai_client.cache_clear()
    assert captured["model"] == "text-embedding-3-small"
    assert captured["dimensions"] == 384
    assert vectors.shape == (1, 384)
