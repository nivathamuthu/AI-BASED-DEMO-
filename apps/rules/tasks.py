"""Run the rules pipeline off the request path."""
from celery import shared_task

from apps.rules.services.pipeline import run_rules


@shared_task
def task_extract_rules(force: bool = False, version: str | None = None) -> dict:
    run = run_rules(version=version, force=force)
    return {"run": run.id, **run.stats["totals"]}
