"""BM25 over the chunk table. The index is rebuilt when the set of chunks changes."""
import re

from rank_bm25 import BM25Okapi

from apps.ingest.models import Chunk
from apps.rag.services.hits import Hit

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_CACHE: dict = {"key": None, "chunks": [], "index": None}


class _BM25(BM25Okapi):
    """BM25Okapi, with a floor so a rare term in a tiny corpus does not score zero.

    The stock idf is log(N - df + 0.5) - log(df + 0.5), which is exactly 0 when a
    term appears in one document out of two. That hid exact part numbers in tests.
    """

    def _calc_idf(self, nd):
        super()._calc_idf(nd)
        for word, value in self.idf.items():
            if value <= 0:
                self.idf[word] = self.epsilon


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


_OVERLAP_STOP = {
    "a", "an", "the", "is", "of", "to", "and", "or", "for", "in", "on", "with",
    "what", "which", "when", "where", "how", "does", "this", "that", "are", "be",
    "by", "from", "at", "as", "it", "its", "was", "were", "can", "not",
}


def content_tokens(text: str) -> set[str]:
    """Tokens that count as evidence a chunk is about the question."""
    return {token for token in tokenize(text) if token not in _OVERLAP_STOP and len(token) >= 3}


def search_bm25(question: str, k: int) -> list[Hit]:
    chunks, index = _index()
    if index is None:
        return []
    scores = index.get_scores(tokenize(question))
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
    hits = []
    for position in order:
        chunk = chunks[position]
        score = float(scores[position])
        hits.append(
            Hit(
                chunk_id=chunk.id,
                text=chunk.text,
                document_id=chunk.document_id,
                document_title=chunk.document.title,
                page_start=chunk.page_start,
                section=chunk.section_path,
                score=round(score, 4),
                lexical=len(content_tokens(question) & content_tokens(chunk.text)),
            )
        )
    return [hit for hit in hits if hit.score > 0]


def _index():
    chunks = list(Chunk.objects.select_related("document").order_by("id"))
    key = (len(chunks), chunks[-1].id if chunks else 0)
    if _CACHE["key"] != key:
        tokenized = [tokenize(chunk.text) for chunk in chunks]
        _CACHE["key"] = key
        _CACHE["chunks"] = chunks
        _CACHE["index"] = _BM25(tokenized) if tokenized else None
    return _CACHE["chunks"], _CACHE["index"]


def clear_cache() -> None:
    """Drop the in-process index. Tests call this after inserting chunks."""
    _CACHE["key"] = None
    _CACHE["chunks"] = []
    _CACHE["index"] = None
