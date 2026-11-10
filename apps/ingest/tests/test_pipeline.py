"""The pipeline end to end, with the mock embedder and Celery running in-process."""
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from apps.ingest.models import Document
from apps.ingest.services.detect import UnsupportedFileType
from apps.ingest.services.intake import ingest_bytes
from core.embeddings import EMBEDDING_DIM

pytestmark = pytest.mark.django_db


def _note(name: str = "note.txt") -> tuple[str, bytes]:
    text = (
        "1. Applicability\n\n"
        "AD 2023-14-09 applies when part 65-45210-3 is installed.\n\n"
        "The replacement bushing is PN-77-4418 and the limit is 12000 flight hours.\n"
    )
    return name, text.encode()


def test_upload_runs_every_stage_and_stores_citations():
    document, outcome = ingest_bytes(*_note())

    assert outcome == "created"
    document.refresh_from_db()
    assert document.status == Document.Status.DONE
    assert document.file_type == "txt"
    run = document.runs.get()
    stages = list(run.stages.order_by("position"))
    assert [stage.name for stage in stages] == [
        "detect", "parse", "clean", "chunk", "enrich", "embed", "index",
    ]
    assert all(stage.status == "done" for stage in stages)
    assert all(stage.attempts == 1 for stage in stages)
    assert sum(stage.duration_ms for stage in stages) >= 0

    chunks = list(document.chunks.order_by("ordinal"))
    assert chunks
    assert all(len(chunk.embedding) == EMBEDDING_DIM for chunk in chunks)
    assert any(chunk.page_start == 1 for chunk in chunks)
    identifiers = [
        item["value"]
        for chunk in chunks
        for item in chunk.metadata["identifiers"]
    ]
    assert "AD 2023-14-09" in identifiers
    assert "PN-77-4418" in identifiers
    assert "65-45210-3" in identifiers


def test_same_bytes_are_not_ingested_twice():
    first, _ = ingest_bytes(*_note())
    second, outcome = ingest_bytes(*_note("other-name.txt"))

    assert outcome == "duplicate"
    assert second.pk == first.pk
    assert Document.objects.count() == 1
    assert first.runs.count() == 1


def test_a_failed_stage_is_retried_and_then_succeeds(settings):
    settings.INGEST_CHAOS_STAGE = "parse"
    document, _ = ingest_bytes(*_note())

    stage = document.runs.get().stages.get(name="parse")
    assert stage.attempts == 2
    assert stage.status == "done"
    assert document.runs.get().status == "done"
    assert document.chunks.exists()


def test_unsupported_extension_is_rejected_before_a_row_is_created():
    with pytest.raises(UnsupportedFileType):
        ingest_bytes("picture.png", b"\x89PNG not really")
    assert Document.objects.count() == 0


def test_upload_page_lists_a_finished_document(client):
    name, data = _note()
    response = client.post("/ingest/", {"file": SimpleUploadedFile(name, data), "title": "Side-stay note"})
    assert response.status_code == 302

    page = client.get(response["Location"])
    body = page.content.decode()
    assert page.status_code == 200
    assert "Side-stay note" in body
    assert "AD 2023-14-09" in body
    assert "embed" in body
