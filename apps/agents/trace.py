"""Per-step collection of tool calls, LLM calls and notes, filled while a node runs."""
from __future__ import annotations

import functools
import time
from contextvars import ContextVar
from dataclasses import dataclass, field

from core.llm import structured_call
from core.models import LLMCall
from core.prompts import render_prompt

PROMPT_VERSION = "v1"


@dataclass
class StepTrace:
    tool_calls: list[dict] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


_CURRENT: ContextVar[StepTrace | None] = ContextVar("agent_step_trace", default=None)


def begin() -> tuple[StepTrace, object]:
    trace = StepTrace()
    return trace, _CURRENT.set(trace)


def end(token) -> None:
    _CURRENT.reset(token)


def note(**values) -> None:
    """Trace-only details for the current step (shown in the UI, not part of the graph state)."""
    trace = _CURRENT.get()
    if trace is not None:
        trace.notes.update(values)


def tool(summarize):
    """Record a tool call: name, typed arguments, a short result summary, duration."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(**kwargs):
            started = time.perf_counter()
            result = fn(**kwargs)
            trace = _CURRENT.get()
            if trace is not None:
                trace.tool_calls.append({
                    "tool": fn.__name__,
                    "args": kwargs,
                    "result": summarize(result),
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                })
            return result

        return wrapper

    return decorator


def ask(schema, prompt_name: str, variables: dict, task: str):
    """Render prompts/<prompt_name>/v1.md and make one validated structured call through core.llm."""
    prompt = render_prompt(prompt_name, PROMPT_VERSION, {"domain": "aviation maintenance", **variables}, schema)
    result = structured_call(schema, prompt, task=task, prompt_name=prompt_name, prompt_version=PROMPT_VERSION)
    call = LLMCall.objects.filter(task=task).order_by("-id").first()
    trace = _CURRENT.get()
    if trace is not None and call is not None:
        trace.llm_calls.append({
            "task": task,
            "schema": schema.__name__,
            "provider": call.provider,
            "model": call.model,
            "attempts": call.attempts,
            "latency_ms": call.latency_ms,
            "tokens": f"{call.input_tokens}/{call.output_tokens}",
        })
    return result
