"""Retrieval fusion, the query page, and re-embedding."""
import pytest
from django.core.files.base import ContentFile
from django.core.management import call_command

from apps.ingest.models import Chunk, Document
from apps.rag.services.answer import answer_question
from apps.rag.services.bm25 import clear_cache
from apps.rag.services.eval import hit_at_k
from apps.rag.services.fusion import rrf
from apps.rag.services.retrieve import retrieve
from core.embeddings import embed_texts

pytestmark = pytest.mark.django_db


def test_rrf_rewards_a_document_that_both_lists_rank_first():
    fused = dict(rrf([[1, 2, 3], [1, 3, 4]]))
    assert fused[1] > fused[2]
    assert fused[1] > fused[3]
    assert abs(fused[1] - (1 / 61 + 1 / 61)) < 1e-9


def test_hit_at_k_checks_the_top_texts_only():
    assert hit_at_k(["nope", "PN-77-4418 is the bushing"], ["PN-77-4418"], k=5) is True
    assert hit_at_k(["nope"], ["PN-77-4418"], k=5) is False


def test_exact_part_number_is_retrieved_and_cited():
    document = _document()
    _chunk(document, "Crew meals are loaded in the forward galley.", 0)
    _chunk(document, "Replace part 65-45210-3 with PN-77-4418 at 12000 flight hours under AD 2023-14-09.", 1)
    clear_cache()

    for mode in ("bm25", "vector", "hybrid", "hybrid_rerank"):
        hits = retrieve("65-45210-3", mode, k=5)
        assert hits, mode
        assert "65-45210-3" in hits[0].text

    for mode in ("hybrid", "vector"):
        result = answer_question("Which part replaces 65-45210-3?", mode)
        assert result["known"] is True, mode
        assert "PN-77-4418" in result["answer"]
        assert result["citations"]
        assert "p.2" in result["citations"][0]


def test_unknown_question_abstains():
    document = _document()
    _chunk(document, "The side-stay is inspected at every C-check.", 0)
    clear_cache()
    result = answer_question("What torque in newton-metres is specified for the pin?", "hybrid")
    assert result["known"] is False
    assert result["citations"] == []


def test_query_page_renders_an_answer(client):
    document = _document()
    _chunk(document, "Compliance is recorded against AD 2023-14-09 for the side-stay.", 0)
    clear_cache()
    response = client.post("/rag/", {"question": "Which directive governs the side-stay?", "mode": "bm25"})
    body = response.content.decode()
    assert response.status_code == 200
    assert "AD 2023-14-09" in body


def test_reembed_records_the_provider():
    document = _document()
    chunk = _chunk(document, "Replace part 65-45210-3 with PN-77-4418.", 0)
    chunk.embedding = [0.0] * 384
    chunk.save(update_fields=["embedding"])
    call_command("reembed")
    chunk.refresh_from_db()
    assert chunk.metadata["embedding_provider"] == "mock"
    assert chunk.metadata["embedding_model"]
    assert len(chunk.embedding) == 384
    assert chunk.embedding[0] != 0.0 or any(value != 0.0 for value in chunk.embedding)


def _document() -> Document:
    document = Document(
        title="Side-stay note",
        original_name="note.txt",
        sha256="a" * 64,
        size_bytes=20,
        file_type="txt",
        status=Document.Status.DONE,
    )
    document.file.save("note.txt", ContentFile(b"note"), save=False)
    document.save()
    return document


def _chunk(document, text, ordinal) -> Chunk:
    return Chunk.objects.create(
        document=document,
        ordinal=ordinal,
        text=text,
        text_hash=f"hash-{ordinal}",
        page_start=ordinal + 1,
        section_title="1. Applicability",
        section_path="1. Applicability",
        token_count=12,
        metadata={},
        embedding=embed_texts([text])[0].tolist(),
    )
