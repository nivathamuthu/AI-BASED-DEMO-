from django.shortcuts import render

from core.models import EvalRun


def compare(request):
    """Side-by-side v1 and v2 outputs from the latest prompt eval."""
    run = EvalRun.objects.filter(kind="prompts").first()
    rows = run.metrics.get("rows", []) if run else []
    return render(request, "prompts/compare.html", {"run": run, "rows": rows})
