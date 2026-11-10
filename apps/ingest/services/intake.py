"""Accept an upload, dedupe it by file hash, and start a pipeline run."""
import hashlib
from pathlib import Path

from django.core.files.base import ContentFile

from apps.ingest.models import Document, PipelineRun
from apps.ingest.services.detect import SUPPORTED, UnsupportedFileType


def ingest_path(path: Path, title: str = "") -> tuple[Document, str]:
    """Ingest a file from disk. Returns (document, outcome).

    outcome is `created` (new document, run started), `duplicate` (same bytes
    already ingested) or `restarted` (same bytes, but the previous run failed).
    """
    return ingest_bytes(path.name, path.read_bytes(), title=title or path.stem)


def ingest_bytes(name: str, data: bytes, title: str = "") -> tuple[Document, str]:
    """Same as `ingest_path`, from an uploaded filename and its bytes."""
    ext = Path(name).suffix.lower().lstrip(".")
    if ext == "text":
        ext = "txt"
    if ext not in SUPPORTED:
        raise UnsupportedFileType(
            f"{name} is not supported. Upload one of: {', '.join(sorted(SUPPORTED))}."
        )
    digest = hashlib.sha256(data).hexdigest()
    existing = Document.objects.filter(sha256=digest).first()
    if existing is not None:
        latest = existing.runs.order_by("-id").first()
        if latest is not None and latest.status == PipelineRun.Status.FAILED:
            from apps.ingest.tasks import start_run

            start_run(existing)
            return existing, "restarted"
        return existing, "duplicate"

    document = Document(
        title=(title or Path(name).stem)[:255],
        original_name=Path(name).name[:255],
        sha256=digest,
        size_bytes=len(data),
        status=Document.Status.QUEUED,
    )
    document.file.save(Path(name).name, ContentFile(data), save=False)
    document.save()
    from apps.ingest.tasks import start_run

    start_run(document)
    return document, "created"
