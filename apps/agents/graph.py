"""The LangGraph compliance assistant, its Postgres checkpointer, and the run / resume entry points.

classify -> plan -> hybrid_retrieve -> graph_lookup -> rule_check -> draft -> verify_citations
    -> human_approval (interrupt) -> final
classify --out_of_scope--> refuse
verify_citations --failed, loops < AGENT_MAX_LOOPS--> hybrid_retrieve
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import TypedDict

from django.conf import settings
from django.db import connection
from django.utils import timezone
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from psycopg.conninfo import make_conninfo

from apps.agents import nodes, trace
from apps.agents.models import AgentRun, AgentStep
from core.llm import describe_backend

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    run_id: int
    question: str
    intent: str
    entities: list[str]
    plan: dict
    query: str
    loops: int
    hits: list[dict]
    facts: list[dict]
    graph_error: str
    rules: list[dict]
    rule_check: dict
    draft: dict
    verification: dict
    retry: bool
    warning: str
    approval: dict
    final: dict


# State keys each node reads, saved as the step input.
READS = {
    "classify": ["question"],
    "refuse": ["intent"],
    "plan": ["question", "intent", "entities"],
    "hybrid_retrieve": ["query", "loops"],
    "graph_lookup": ["question", "query", "plan"],
    "rule_check": ["question", "entities", "plan"],
    "draft": ["question", "hits", "facts", "rule_check", "loops"],
    "verify_citations": ["draft", "hits", "loops"],
    "human_approval": ["draft", "warning"],
    "final": ["approval", "verification"],
}


def _brief(key: str, value):
    """Keep the trace readable: sources become citations, long text is cut."""
    if key == "hits" and isinstance(value, list):
        return [f"S{index} #{row['chunk_id']} {row['citation']} (via {row['origin']}, loop {row.get('loop', 0)})"
                for index, row in enumerate(value, start=1)]
    if key == "facts" and isinstance(value, list):
        return [f"{fact['source']} {fact['relation']} {fact['target']} - {fact['citation']}" for fact in value]
    if key == "rules" and isinstance(value, list):
        return [f"{row['key']} ({row['status']}): IF {row['condition']} THEN {row['action']}" for row in value]
    if isinstance(value, str) and len(value) > 1500:
        return value[:1500] + "..."
    return value


def _record(name: str, state: dict, delta: dict, step: trace.StepTrace, status: str, error: str, ms: int) -> None:
    run_id = state["run_id"]
    if name == "human_approval" and status == AgentStep.Status.DONE:
        waiting = AgentStep.objects.filter(run_id=run_id, node=name, status=AgentStep.Status.WAITING).last()
        if waiting is not None:
            waited = int((timezone.now() - waiting.created_at).total_seconds() * 1000)
            waiting.status = status
            waiting.output = {key: _brief(key, value) for key, value in delta.items()}
            waiting.duration_ms = waited
            waiting.save(update_fields=["status", "output", "duration_ms"])
            return
    output = {key: _brief(key, value) for key, value in delta.items()}
    if step.notes:
        output["notes"] = step.notes
    AgentStep.objects.create(
        run_id=run_id,
        order=AgentStep.objects.filter(run_id=run_id).count() + 1,
        node=name,
        loop=delta.get("loops", state.get("loops", 0)) if name != "verify_citations" else state.get("loops", 0),
        status=status,
        input={key: _brief(key, state.get(key)) for key in READS.get(name, []) if key in state},
        output=output,
        tool_calls=step.tool_calls,
        llm_calls=step.llm_calls,
        duration_ms=ms,
        error=error,
    )
    updates = {}
    if "intent" in delta:
        updates["intent"] = delta["intent"]
    if "loops" in delta:
        updates["loops"] = delta["loops"]
    if updates:
        AgentRun.objects.filter(id=run_id).update(**updates, updated_at=timezone.now())


def traced(name: str, fn):
    def node(state):
        step, token = trace.begin()
        started = time.perf_counter()
        delta, status, error = {}, AgentStep.Status.DONE, ""
        try:
            delta = fn(state) or {}
            return delta
        except GraphInterrupt:
            status = AgentStep.Status.WAITING
            raise
        except Exception as exc:
            status, error = AgentStep.Status.FAILED, f"{type(exc).__name__}: {exc}"
            raise
        finally:
            trace.end(token)
            _record(name, state, delta, step, status, error, int((time.perf_counter() - started) * 1000))

    node.__name__ = name
    return node


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    for name in ("classify", "refuse", "plan", "hybrid_retrieve", "graph_lookup", "rule_check", "draft",
                 "verify_citations", "human_approval", "final"):
        graph.add_node(name, traced(name, getattr(nodes, name)))
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", nodes.route_after_classify, {"refuse": "refuse", "plan": "plan"})
    graph.add_edge("refuse", END)
    graph.add_edge("plan", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "graph_lookup")
    graph.add_edge("graph_lookup", "rule_check")
    graph.add_edge("rule_check", "draft")
    graph.add_edge("draft", "verify_citations")
    graph.add_conditional_edges("verify_citations", nodes.route_after_verify,
                                {"hybrid_retrieve": "hybrid_retrieve", "human_approval": "human_approval"})
    graph.add_edge("human_approval", "final")
    graph.add_edge("final", END)
    return graph


_SET_UP: set[str] = set()


def _conninfo() -> str:
    """The Django database (the test database under pytest), as a libpq connection string."""
    db = connection.settings_dict
    return make_conninfo(host=db["HOST"], port=str(db["PORT"] or 5432), dbname=db["NAME"],
                         user=db["USER"], password=db["PASSWORD"])


@contextmanager
def compiled_graph():
    """The graph compiled with a Postgres checkpointer on its own connection, closed afterwards."""
    conninfo = _conninfo()
    with PostgresSaver.from_conn_string(conninfo) as saver:
        if conninfo not in _SET_UP:
            saver.setup()
            _SET_UP.add(conninfo)
        yield build_graph().compile(checkpointer=saver)


def _config(run: AgentRun) -> dict:
    return {"configurable": {"thread_id": run.thread_id}, "recursion_limit": 60}


def create_run(question: str, source: str = "ui") -> AgentRun:
    backend = describe_backend()
    return AgentRun.objects.create(
        question=question.strip(), thread_id=uuid.uuid4().hex, source=source,
        llm_provider=backend["provider"], llm_model=backend["model"], retrieval_mode=settings.AGENT_RETRIEVAL_MODE,
    )


def _sync(run: AgentRun, snapshot) -> AgentRun:
    values = snapshot.values
    run.intent = values.get("intent", run.intent)
    run.loops = values.get("loops", run.loops)
    run.draft = values.get("draft", run.draft)
    run.warning = values.get("warning", "")
    if values.get("verification"):
        run.verified = values["verification"]["passed"]
        run.review = {
            "conflicts": (values.get("rule_check") or {}).get("conflicts", []),
            "problems": values["verification"]["problems"],
            "checks": values["verification"]["checks"],
            "sources": [{"n": index, "citation": row["citation"], "origin": row["origin"]}
                        for index, row in enumerate(values.get("hits", []), start=1)],
        }
    if snapshot.next:
        run.status = AgentRun.Status.AWAITING_APPROVAL
    else:
        run.final = values.get("final", {})
        run.status = {"refused": AgentRun.Status.REFUSED, "approved": AgentRun.Status.DONE,
                      "rejected": AgentRun.Status.REJECTED}.get(run.final.get("status"), AgentRun.Status.FAILED)
        run.finished_at = timezone.now()
    run.save()
    return run


def _fail(run: AgentRun, exc: Exception) -> None:
    run.status = AgentRun.Status.FAILED
    run.error = f"{type(exc).__name__}: {exc}"
    run.finished_at = timezone.now()
    run.save(update_fields=["status", "error", "finished_at", "updated_at"])
    logger.exception("agent run %s failed", run.id)


def execute(run: AgentRun) -> AgentRun:
    """Run from classify until the approval interrupt (or the refusal)."""
    run.status = AgentRun.Status.RUNNING
    run.save(update_fields=["status", "updated_at"])
    try:
        with compiled_graph() as app:
            app.invoke({"run_id": run.id, "question": run.question}, _config(run))
            return _sync(run, app.get_state(_config(run)))
    except Exception as exc:
        _fail(run, exc)
        raise


def resume(run: AgentRun, approved: bool, note: str = "") -> AgentRun:
    """Resume the paused thread with the reviewer's decision."""
    decided_in_ui = run.status == AgentRun.Status.RUNNING and run.approved is not None
    if run.status != AgentRun.Status.AWAITING_APPROVAL and not decided_in_ui:
        raise ValueError(f"run {run.id} is {run.status}, not awaiting approval")
    run.approved, run.approval_note, run.status = approved, note, AgentRun.Status.RUNNING
    run.save(update_fields=["approved", "approval_note", "status", "updated_at"])
    try:
        with compiled_graph() as app:
            app.invoke(Command(resume={"approved": approved, "note": note}), _config(run))
            return _sync(run, app.get_state(_config(run)))
    except Exception as exc:
        _fail(run, exc)
        raise


def checkpoint_count(run: AgentRun) -> int:
    with compiled_graph() as app:
        return sum(1 for _ in app.checkpointer.list(_config(run)))
