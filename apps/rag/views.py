from django.shortcuts import render

from apps.rag.services.answer import answer_question
from apps.rag.services.retrieve import MODES
from core.models import EvalRun


def query(request):
    result = None
    mode = request.GET.get("mode", "hybrid")
    if request.method == "POST":
        mode = request.POST.get("mode", "hybrid")
        question = (request.POST.get("question") or "").strip()
        if question:
            if mode not in MODES:
                mode = "hybrid"
            result = answer_question(question, mode)
    latest = EvalRun.objects.filter(kind="rag").first()
    return render(
        request,
        "rag/query.html",
        {"result": result, "mode": mode, "modes": MODES, "latest_eval": latest},
    )
