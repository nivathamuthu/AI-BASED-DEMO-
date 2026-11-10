"""Answer a question from retrieved chunks, and say so when the context is too weak."""
from apps.prompts.services import run_prompt
from apps.rag.services.hits import Hit
from apps.rag.services.retrieve import is_weak, retrieve

UNKNOWN = "I don't know. The retrieved context is not strong enough to answer."


def answer_question(question: str, mode: str, k: int = 5) -> dict:
    """Retrieve, then ask the v2 citation prompt. Abstain when retrieval or the model is unsure."""
    hits = retrieve(question, mode, k=k)
    if is_weak(hits, mode):
        return _payload(question, mode, UNKNOWN, known=False, hits=hits)
    numbered = "\n\n".join(
        f"[{index}] {hit.text}\n(source: {hit.citation()})" for index, hit in enumerate(hits, start=1)
    )
    output = run_prompt(
        "answer_with_citations",
        "v2",
        {"chunk": numbered, "question": question, "domain": "aviation maintenance"},
    )
    if not output.known or output.answer.strip().lower() == "unknown":
        return _payload(question, mode, UNKNOWN, known=False, hits=hits)
    cited = _cited(hits, output.citation_indexes)
    return _payload(question, mode, output.answer, known=True, hits=cited or hits)


def _cited(hits: list[Hit], indexes: list[int]) -> list[Hit]:
    chosen = []
    for index in indexes:
        if 1 <= index <= len(hits):
            chosen.append(hits[index - 1])
    return chosen


def _payload(question: str, mode: str, answer: str, known: bool, hits: list[Hit]) -> dict:
    return {
        "question": question,
        "mode": mode,
        "answer": answer,
        "known": known,
        "citations": [hit.citation() for hit in hits] if known else [],
        "hits": [
            {
                "chunk_id": hit.chunk_id,
                "document": hit.document_title,
                "page": hit.page_start,
                "section": hit.section,
                "score": hit.score,
                "text": hit.text,
            }
            for hit in hits
        ],
    }
