"""Rerank a fused list.

The default is a lexical overlap sort, which needs no model. Set RERANK_ENABLED=true
to use a cross-encoder. That import exists only in an image built with INSTALL_ML=1.
"""
import logging
from functools import lru_cache

from django.conf import settings

from apps.rag.services.bm25 import tokenize
from apps.rag.services.hits import Hit

logger = logging.getLogger(__name__)


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    """Return `hits` best-first. Falls back to lexical overlap when the cross-encoder is unavailable."""
    if settings.RERANK_ENABLED:
        try:
            return _cross_encoder(query, hits)
        except ImportError:
            logger.warning(
                "RERANK_ENABLED is set but sentence-transformers is not installed. "
                "Using the lexical rerank. Rebuild with --build-arg INSTALL_ML=1 to enable the cross-encoder."
            )
    return _lexical(query, hits)


def _lexical(query: str, hits: list[Hit]) -> list[Hit]:
    wanted = set(tokenize(query))
    ranked = sorted(hits, key=lambda hit: len(wanted & set(tokenize(hit.text))), reverse=True)
    for hit in ranked:
        hit.score = float(len(wanted & set(tokenize(hit.text))))
        hit.lexical = int(hit.score)
    return ranked


def _cross_encoder(query: str, hits: list[Hit]) -> list[Hit]:
    model = _encoder(settings.RERANK_MODEL)
    scores = model.predict([(query, hit.text) for hit in hits])
    ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
    ordered = []
    for hit, score in ranked:
        hit.score = round(float(score), 4)
        ordered.append(hit)
    return ordered


@lru_cache(maxsize=1)
def _encoder(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)
