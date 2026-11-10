"""One Celery task per stage. A failing stage is retried, then the run is marked failed.

In tests (CELERY_TASK_ALWAYS_EAGER) the retry happens in-process, so a test can
run the whole pipeline with no worker. In Docker the retry is a real Celery retry,
which shows up in the worker log and as `attempts` on the status page.
"""
import logging

from celery import shared_task
from django.conf import settings
from django.utils import timezone

from apps.ingest.constants import STAGES
from apps.ingest.models import Document, PipelineRun, PipelineStage
from apps.ingest.services.stages import STAGE_FUNCTIONS
from core.timing import timed

logger = logging.getLogger(__name__)


def start_run(document: Document) -> PipelineRun:
    """Open a run with one pending row per stage, and enqueue `detect`."""
    document.status = Document.Status.QUEUED
    document.save(update_fields=["status", "updated_at"])
    run = PipelineRun.objects.create(document=document, status=PipelineRun.Status.QUEUED)
    PipelineStage.objects.bulk_create(
        [PipelineStage(run=run, name=name, position=position) for position, name in enumerate(STAGES)]
    )
    _enqueue(task_detect, run.id)
    return run


def _enqueue(task, run_id: int) -> None:
    if settings.CELERY_TASK_ALWAYS_EAGER:
        task.apply(args=[run_id])
    else:
        task.delay(run_id)


def _drive(task, run_id: int, stage_name: str, next_task) -> None:
    run = PipelineRun.objects.select_related("document").get(id=run_id)
    if run.status == PipelineRun.Status.FAILED:
        return
    outcome = _attempt(run, stage_name)
    if outcome == "retry":
        if settings.CELERY_TASK_ALWAYS_EAGER:
            _drive(task, run_id, stage_name, next_task)
            return
        raise task.retry(countdown=1)
    if outcome == "ok" and next_task is not None:
        _enqueue(next_task, run_id)


def _attempt(run: PipelineRun, stage_name: str) -> str:
    """Run one attempt. Returns `ok`, `retry` or `failed`."""
    stage = run.stages.get(name=stage_name)
    document = run.document
    stage.attempts += 1
    stage.status = PipelineStage.Status.RUNNING
    stage.started_at = timezone.now()
    stage.error = ""
    stage.save()
    run.status = PipelineRun.Status.RUNNING
    run.current_stage = stage_name
    run.save(update_fields=["status", "current_stage"])
    document.status = Document.Status.RUNNING
    document.save(update_fields=["status", "updated_at"])

    try:
        if settings.INGEST_CHAOS_STAGE == stage_name and stage.attempts == 1:
            raise RuntimeError(f"Injected failure of stage {stage_name!r} on attempt 1 (INGEST_CHAOS_STAGE).")
        with timed() as timer:
            summary = STAGE_FUNCTIONS[stage_name](document)
    except Exception as exc:
        logger.warning("stage %s attempt %s failed: %s", stage_name, stage.attempts, exc)
        stage.error = str(exc)[:2000]
        stage.finished_at = timezone.now()
        if stage.attempts <= settings.INGEST_MAX_RETRIES:
            stage.status = PipelineStage.Status.RETRYING
            stage.save()
            return "retry"
        stage.status = PipelineStage.Status.FAILED
        stage.save()
        run.status = PipelineRun.Status.FAILED
        run.error = stage.error
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "error", "finished_at"])
        document.status = Document.Status.FAILED
        document.save(update_fields=["status", "updated_at"])
        return "failed"

    stage.status = PipelineStage.Status.DONE
    stage.duration_ms = timer.duration_ms
    stage.finished_at = timezone.now()
    stage.summary = summary
    stage.save()
    logger.info("stage %s done in %sms %s", stage_name, timer.duration_ms, summary)
    if stage_name == STAGES[-1]:
        run.status = PipelineRun.Status.DONE
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "finished_at"])
        document.status = Document.Status.DONE
        document.save(update_fields=["status", "updated_at"])
        _after_ingest(document.id)
    return "ok"


def _after_ingest(document_id: int) -> None:
    """Hand the finished document to the knowledge graph. A KG failure never fails the ingest run."""
    if not settings.KG_AUTO_BUILD:
        return
    from apps.kg.tasks import task_kg_document

    try:
        if settings.CELERY_TASK_ALWAYS_EAGER:
            task_kg_document.apply(args=[document_id])
        else:
            task_kg_document.delay(document_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("kg hand-off for document %s failed: %s", document_id, exc)


@shared_task(bind=True, max_retries=10)
def task_detect(self, run_id: int) -> None:
    _drive(self, run_id, "detect", task_parse)


@shared_task(bind=True, max_retries=10)
def task_parse(self, run_id: int) -> None:
    _drive(self, run_id, "parse", task_clean)


@shared_task(bind=True, max_retries=10)
def task_clean(self, run_id: int) -> None:
    _drive(self, run_id, "clean", task_chunk)


@shared_task(bind=True, max_retries=10)
def task_chunk(self, run_id: int) -> None:
    _drive(self, run_id, "chunk", task_enrich)


@shared_task(bind=True, max_retries=10)
def task_enrich(self, run_id: int) -> None:
    _drive(self, run_id, "enrich", task_embed)


@shared_task(bind=True, max_retries=10)
def task_embed(self, run_id: int) -> None:
    _drive(self, run_id, "embed", task_index)


@shared_task(bind=True, max_retries=10)
def task_index(self, run_id: int) -> None:
    _drive(self, run_id, "index", None)
