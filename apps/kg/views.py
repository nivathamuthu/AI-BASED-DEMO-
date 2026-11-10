from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_sameorigin

from apps.ingest.models import Chunk
from apps.kg.models import GraphBuild
from apps.kg.schemas import ENTITY_TYPES
from apps.kg.services.graphrag import graph_answer
from apps.kg.services.viz import graph_html
from apps.rag.services.answer import answer_question
from core.graph_store import GraphStoreError, get_store
from core.models import EvalRun


def index(request):
    """Graph, counts, type filter and the GraphRAG question box."""
    selected = [kind for kind in request.GET.getlist("type") if kind in ENTITY_TYPES]
    context = {
        "build": GraphBuild.objects.first(),
        "types": ENTITY_TYPES,
        "selected": selected,
        "type_query": "&".join(f"type={kind}" for kind in selected),
        "latest_eval": EvalRun.objects.filter(kind="kg").first(),
        "error": "",
        "result": None,
        "plain": None,
    }
    if request.method == "POST":
        question = (request.POST.get("question") or "").strip()
        if question:
            try:
                context["result"] = graph_answer(question)
                context["plain"] = answer_question(question, "hybrid")
            except GraphStoreError as exc:
                context["error"] = str(exc)
    return render(request, "kg/index.html", context)


@xframe_options_sameorigin
def graph_frame(request):
    selected = {kind for kind in request.GET.getlist("type") if kind in ENTITY_TYPES}
    try:
        store = get_store()
        html = graph_html(store.nodes(), store.edges(), selected or None)
    except GraphStoreError as exc:
        return HttpResponse(f"<p style='font-family:sans-serif'>{exc}</p>", status=503)
    return HttpResponse(html)


def node_detail(request):
    """Source chunks, with citations, for one clicked node."""
    node_id = request.GET.get("id", "")
    try:
        store = get_store()
        node = next((row for row in store.nodes() if row["id"] == node_id), None)
        edges = [edge for edge in store.edges() if node_id in (edge["source"], edge["target"])] if node else []
        names = {row["id"]: row["name"] for row in store.nodes()} if node else {}
    except GraphStoreError as exc:
        return HttpResponse(str(exc), status=503)
    if node is None:
        return HttpResponse("<p class='muted'>Unknown node.</p>")
    chunk_ids = [row["chunk_id"] for row in node.get("provenance", [])]
    chunks = {chunk.id: chunk for chunk in Chunk.objects.select_related("document").filter(id__in=chunk_ids)}
    sources = []
    for row in node.get("provenance", []):
        chunk = chunks.get(row["chunk_id"])
        if chunk:
            sources.append({**row, "text": chunk.text})
    relations = [
        {"source": names.get(edge["source"], edge["source"]), "relation": edge["relation"],
         "target": names.get(edge["target"], edge["target"]), "evidence": edge.get("evidence", "")}
        for edge in edges
    ]
    return render(request, "kg/_node.html", {"node": node, "sources": sources, "relations": relations})
