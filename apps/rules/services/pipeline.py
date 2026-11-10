"""Extract -> verify quotes -> sync keys -> merge -> conflicts, one pass at a time."""
import logging
from contextlib import contextmanager

from django.conf import settings

from apps.ingest.models import Chunk, Document
from apps.rules.models import Rule, RuleConflict, RuleExtraction, RuleRun
from apps.rules.services.conflicts import detect_conflicts
from apps.rules.services.extract import extract_chunks
from apps.rules.services.merge import merge_duplicates
from core.embeddings import describe as describe_embeddings
from core.llm import describe_backend

logger = logging.getLogger(__name__)


@contextmanager
def _run_lock():
    """Key allocation and merging assume one pass at a time across web, worker and commands."""
    if settings.CELERY_TASK_ALWAYS_EAGER:
        yield
        return
    import redis

    try:
        lock = redis.Redis.from_url(settings.CELERY_BROKER_URL).lock("docintel:rules-run", timeout=1800, blocking_timeout=900)
        acquired = lock.acquire()
    except redis.RedisError as exc:
        logger.warning("rules lock unavailable, running without it: %s", exc)
        yield
        return
    if not acquired:
        raise RuntimeError("Another rules extraction is still running.")
    try:
        yield
    finally:
        try:
            lock.release()
        except redis.RedisError:
            pass


def run_rules(document_ids: list[int] | None = None, version: str | None = None, force: bool = False) -> RuleRun:
    version = version or settings.RULES_PROMPT_VERSION
    documents = Document.objects.filter(status=Document.Status.DONE)
    if document_ids:
        documents = documents.filter(id__in=document_ids)
    chunks = Chunk.objects.filter(document__in=documents).select_related("document").order_by("document_id", "ordinal")
    with _run_lock():
        extraction = extract_chunks(chunks, version=version, force=force)
        merge_log = merge_duplicates()
        conflicts = detect_conflicts()
        backend = describe_backend()
        embeddings = describe_embeddings()
        run = RuleRun.objects.create(
            prompt_version=version,
            llm_provider=backend["provider"],
            llm_model=backend["model"],
            embeddings_provider=embeddings["provider"],
            embeddings_model=embeddings["model"],
            forced=force,
            stats={"extraction": extraction, "merged": len(merge_log), "conflicts": conflicts, "totals": totals(version)},
            merge_log=merge_log,
        )
    return run


def totals(version: str | None = None) -> dict:
    """The review counts: extracted, dropped (quote not found), merged, conflicts, approved."""
    version = version or settings.RULES_PROMPT_VERSION
    extractions = RuleExtraction.objects.filter(prompt_version=version)
    dropped = sum(len(row.dropped) for row in extractions.only("dropped"))
    kept = sum(row.kept_count for row in extractions.only("kept_count"))
    by_status = {status: Rule.objects.filter(status=status).count() for status in Rule.Status.values}
    return {
        "extracted": kept + dropped,
        "kept": kept,
        "dropped": dropped,
        "merged": by_status[Rule.Status.MERGED],
        "conflicts": RuleConflict.objects.filter(is_conflict=True).count(),
        "approved": by_status[Rule.Status.APPROVED],
        "pending": by_status[Rule.Status.PENDING],
        "rejected": by_status[Rule.Status.REJECTED],
        "retired": by_status[Rule.Status.RETIRED],
    }
