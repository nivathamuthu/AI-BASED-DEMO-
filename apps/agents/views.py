import json

from django.conf import settings
from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from apps.agents.graph import create_run
from apps.agents.models import AgentRun
from apps.agents.tasks import task_resume_agent, task_run_agent
from core.embeddings import describe as describe_embeddings
from core.llm import describe_backend
from core.models import EvalRun

EXAMPLES = [
    "Part 65-45210-3 is at 11,800 flight hours on a narrow-body aircraft. Is it due for replacement now?",
    "Can the side-stay replacement on part 65-45210-3 be deferred under the MEL?",
    "Which replacement part number goes with AD 2023-14-09 on the side-stay?",
]


def _context() -> dict:
    return {
        "llm": describe_backend(),
        "embeddings": describe_embeddings(),
        "retrieval_mode": settings.AGENT_RETRIEVAL_MODE,
        "max_loops": settings.AGENT_MAX_LOOPS,
    }


def index(request):
    if request.method == "POST":
        question = (request.POST.get("question") or "").strip()
        if not question:
            messages.error(request, "Type a question.")
            return redirect("agents:index")
        run = create_run(question)
        _dispatch(task_run_agent, run.id)
        return redirect("agents:run", run_id=run.id)
    return render(request, "agents/index.html", {
        **_context(),
        "runs": AgentRun.objects.all()[:15],
        "examples": EXAMPLES,
        "latest_eval": EvalRun.objects.filter(kind="agent").first(),
    })


def _dispatch(task, *args, **kwargs) -> None:
    if settings.CELERY_TASK_ALWAYS_EAGER:
        task.apply(args=args, kwargs=kwargs)
    else:
        task.delay(*args, **kwargs)


def _trace_context(run: AgentRun) -> dict:
    steps = list(run.steps.all())
    for step in steps:
        step.input_json = json.dumps(step.input, indent=2, ensure_ascii=False)
        step.output_json = json.dumps(step.output, indent=2, ensure_ascii=False)
    return {**_context(), "run": run, "steps": steps}


def run_detail(request, run_id: int):
    run = get_object_or_404(AgentRun, id=run_id)
    return render(request, "agents/run.html", _trace_context(run))


def run_trace(request, run_id: int):
    run = get_object_or_404(AgentRun, id=run_id)
    response = render(request, "agents/_trace.html", _trace_context(run))
    if request.GET.get("poll") and not run.active:
        response["HX-Refresh"] = "true"
    return response


@require_POST
def decide(request, run_id: int):
    run = get_object_or_404(AgentRun, id=run_id)
    if run.status != AgentRun.Status.AWAITING_APPROVAL:
        messages.error(request, f"Run {run.id} is {run.get_status_display().lower()}, not awaiting approval.")
        return redirect("agents:run", run_id=run.id)
    approved = request.POST.get("decision") == "approve"
    note = (request.POST.get("note") or "").strip()
    if not approved and not note:
        messages.error(request, "A rejection needs a note.")
        return redirect("agents:run", run_id=run.id)
    AgentRun.objects.filter(id=run.id).update(status=AgentRun.Status.RUNNING, approved=approved, approval_note=note)
    _dispatch(task_resume_agent, run.id, approved, note)
    messages.info(request, "Approved. Resuming the graph." if approved else "Rejected. Resuming the graph.")
    return redirect("agents:run", run_id=run.id)
