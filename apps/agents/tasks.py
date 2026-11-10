"""Run and resume agent threads on the worker, so the page can poll the trace while it runs."""
import logging

from celery import shared_task

from apps.agents import graph
from apps.agents.models import AgentRun

logger = logging.getLogger(__name__)


@shared_task
def task_run_agent(run_id: int) -> str:
    run = AgentRun.objects.get(id=run_id)
    try:
        return graph.execute(run).status
    except Exception as exc:  # the run row already holds the error for the page
        logger.warning("agent run %s failed: %s", run_id, exc)
        return AgentRun.Status.FAILED


@shared_task
def task_resume_agent(run_id: int, approved: bool, note: str = "") -> str:
    run = AgentRun.objects.get(id=run_id)
    try:
        return graph.resume(run, approved=approved, note=note).status
    except Exception as exc:
        logger.warning("agent run %s resume failed: %s", run_id, exc)
        return AgentRun.objects.get(id=run_id).status
