"""Upload form, pipeline status, and the chunk table. The pipeline itself is in services/ and tasks.py."""
from django.contrib import messages
from django.db.models import Count, Prefetch
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from apps.ingest.models import Document, PipelineRun, PipelineStage
from apps.ingest.services.detect import UnsupportedFileType
from apps.ingest.services.intake import ingest_bytes
from apps.ingest.tasks import start_run

ACTIVE = {Document.Status.QUEUED, Document.Status.RUNNING}


def index(request):
    if request.method == "POST":
        upload = request.FILES.get("file")
        if upload is None:
            messages.error(request, "Choose a file first.")
            return redirect("ingest:index")
        title = (request.POST.get("title") or "").strip()
        try:
            document, outcome = ingest_bytes(upload.name, upload.read(), title=title)
        except UnsupportedFileType as exc:
            messages.error(request, str(exc))
            return redirect("ingest:index")
        if outcome == "created":
            messages.success(request, f"Queued {document.original_name}.")
        elif outcome == "restarted":
            messages.warning(request, "That file had failed earlier. It was queued again.")
        else:
            messages.info(request, "Already ingested. Same file hash, so no second copy was created.")
        return redirect("ingest:detail", pk=document.pk)

    documents = _documents()
    return render(request, "ingest/index.html", {"documents": documents, "active": _any_active(documents)})


def detail(request, pk):
    document = _document(pk)
    template = "ingest/_detail.html" if request.htmx else "ingest/detail.html"
    return render(request, template, {"document": document})


@require_POST
def reprocess(request, pk):
    document = get_object_or_404(Document, pk=pk)
    if document.status in ACTIVE:
        messages.info(request, "That document is already running.")
    else:
        start_run(document)
        messages.success(request, "Reprocessing. The previous chunks will be replaced.")
    return redirect("ingest:detail", pk=document.pk)


def _documents():
    documents = list(
        Document.objects.annotate(chunk_count=Count("chunks")).order_by("-created_at")
    )
    runs = (
        PipelineRun.objects.filter(document__in=documents)
        .prefetch_related("stages")
        .order_by("id")
    )
    latest = {}
    for run in runs:
        latest[run.document_id] = run
    for document in documents:
        document.current_run = _decorate(latest.get(document.id))
    return documents


def _document(pk):
    document = get_object_or_404(
        Document.objects.annotate(chunk_count=Count("chunks")).prefetch_related(
            Prefetch("chunks", to_attr="chunk_list"),
            Prefetch(
                "runs",
                queryset=PipelineRun.objects.prefetch_related(
                    Prefetch("stages", queryset=PipelineStage.objects.order_by("position"))
                ).order_by("-id"),
                to_attr="run_list",
            ),
        ),
        pk=pk,
    )
    document.current_run = _decorate(document.run_list[0] if document.run_list else None)
    return document


def _decorate(run):
    """Attach the stage rows and their total time, so the template does not re-query."""
    if run is None:
        return None
    run.stage_list = list(run.stages.all())
    run.total_ms = sum(stage.duration_ms for stage in run.stage_list)
    return run


def _any_active(documents) -> bool:
    return any(document.status in ACTIVE for document in documents)
