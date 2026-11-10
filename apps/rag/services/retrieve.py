"""Hybrid retrieval: BM25, pgvector, reciprocal rank fusion, and an optional rerank."""
from django.conf import settings
from pgvector.django import CosineDistance

from apps.ingest.models import Chunk
from apps.rag.services.bm25 import content_tokens, search_bm25
from apps.rag.services.fusion import rrf
from apps.rag.services.hits import Hit
from apps.rag.services.rerank import rerank
from core.embeddings import embed_query

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def retrieve(question: str, mode: str, k: int = 5) -> list[Hit]:
    """Return the top `k` chunks for one of vector, bm25, hybrid, hybrid_rerank."""
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}")
    pool = 20 if mode == "hybrid_rerank" else k
    if mode == "vector":
        hits = _vector(question, pool)
    elif mode == "bm25":
        hits = search_bm25(question, pool)
    else:
        hits = _fuse(question, pool)
        if mode == "hybrid_rerank":
            hits = rerank(question, hits)
    return hits[:k]


def is_weak(hits: list[Hit], mode: str) -> bool:
    """True when the top hit is too thin to answer from.

    Mock embeddings are a lexical hash, so their cosine is not on the same scale
    as a real encoder. With the mock provider, abstain only when the top chunk
    shares no content token with the question. A real provider keeps the cosine floor.
    """
    if not hits:
        return True
    top = hits[0]
    if mode == "bm25":
        return top.score <= 0
    if settings.EMBEDDINGS_PROVIDER == "mock":
        return top.lexical == 0
    if mode == "vector":
        return top.score < settings.RAG_MIN_VECTOR_SIMILARITY
    return top.lexical == 0 and top.score < settings.RAG_MIN_VECTOR_SIMILARITY


def _vector(question: str, k: int) -> list[Hit]:
    vector = embed_query(question).tolist()
    rows = (
        Chunk.objects.exclude(embedding=None)
        .select_related("document")
        .annotate(distance=CosineDistance("embedding", vector))
        .order_by("distance")[:k]
    )
    hits = []
    for row in rows:
        similarity = 1.0 - float(row.distance)
        hits.append(_hit(row, score=similarity, lexical=_overlap(question, row.text)))
    return hits


def _fuse(question: str, k: int) -> list[Hit]:
    """Reciprocal rank fusion of the vector list and the BM25 list."""
    vector_hits = _vector(question, k)
    lexical_hits = search_bm25(question, k)
    by_id = {hit.chunk_id: hit for hit in vector_hits}
    for hit in lexical_hits:
        by_id.setdefault(hit.chunk_id, hit)
    fused = rrf([
        [hit.chunk_id for hit in vector_hits],
        [hit.chunk_id for hit in lexical_hits],
    ])
    hits = []
    for chunk_id, score in fused[:k]:
        hit = by_id[chunk_id]
        hit.score = score
        hits.append(hit)
    return hits


def _hit(chunk: Chunk, score: float, lexical: int) -> Hit:
    return Hit(
        chunk_id=chunk.id,
        text=chunk.text,
        document_id=chunk.document_id,
        document_title=chunk.document.title,
        page_start=chunk.page_start,
        section=chunk.section_path,
        score=round(score, 4),
        lexical=lexical,
    )


def _overlap(query: str, text: str) -> int:
    return len(content_tokens(query) & content_tokens(text))
