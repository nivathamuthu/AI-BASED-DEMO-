"""Text embeddings, always 384-dim so the pgvector column never changes.

Providers (EMBEDDINGS_PROVIDER):
  local  - sentence-transformers on CPU (requirements/ml.txt): all-MiniLM-L6-v2 by default,
           or any 384-dim model in EMBEDDINGS_MODEL such as BAAI/bge-small-en-v1.5
  openai - text-embedding-3-small with dimensions=384
  mock   - hashed bag-of-words; deterministic, offline, good enough for tests
All vectors are L2-normalised, so cosine distance == 1 - dot product.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np
from django.conf import settings

EMBEDDING_DIM = 384

DEFAULT_MODELS = {
    "local": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
    "mock": "mock-hash-bow-384",
}

_TOKEN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def describe() -> dict:
    """Provider, model and dimension. Does not call the provider."""
    return {"provider": settings.EMBEDDINGS_PROVIDER, "model": model_name(), "dimensions": EMBEDDING_DIM}


def model_name() -> str:
    provider = settings.EMBEDDINGS_PROVIDER
    return settings.EMBEDDINGS_MODEL or DEFAULT_MODELS[provider]


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed many texts. Returns float32 array of shape (len(texts), 384)."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    provider = settings.EMBEDDINGS_PROVIDER
    if provider == "mock":
        vectors = np.vstack([_mock_vector(t) for t in texts])
    elif provider == "local":
        model = _local_model(model_name())
        vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        if vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"{model_name()} returns {vectors.shape[1]}-dim vectors; the column is {EMBEDDING_DIM}-dim")
    elif provider == "openai":
        vectors = np.array(_openai_client(model_name()).embed_documents(texts))
        vectors = _normalise(vectors)
    else:
        raise ValueError(f"Unknown EMBEDDINGS_PROVIDER {provider!r}; use local, openai or mock")
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Returns shape (384,)."""
    return embed_texts([text])[0]


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def _mock_vector(text: str) -> np.ndarray:
    """Signed feature hashing of lowercase tokens and token bigrams."""
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    tokens = _TOKEN.findall(text.lower())
    features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    for feature in features:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % EMBEDDING_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[index] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


@lru_cache(maxsize=1)
def _local_model(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "EMBEDDINGS_PROVIDER=local needs requirements/ml.txt "
            "(uv pip install -r requirements/ml.txt), or set EMBEDDINGS_PROVIDER=mock"
        ) from exc
    return SentenceTransformer(name, device="cpu")


@lru_cache(maxsize=1)
def _openai_client(name: str):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=name, dimensions=EMBEDDING_DIM)
