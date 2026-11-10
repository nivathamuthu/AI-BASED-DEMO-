"""Load the demo corpus, or wipe derived data and rebuild it.

    python scripts/seed_demo.py
        Ingest every file under data/corpus. Same file hash is a no-op.

    python scripts/seed_demo.py --reset
        Wipe documents, rules, the graph and agent runs, then ingest the four
        corpus documents, extract rules and build the graph. Uses the configured
        providers and refuses the mock. EvalRun rows are kept.

The Celery worker must be up. A pipeline failure, a validation failure or an
empty rebuild exits non-zero.
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

SUFFIXES = {".pdf", ".docx", ".pptx", ".txt", ".md"}
INGEST_TIMEOUT_SECONDS = 600


def main() -> None:
    parser = argparse.ArgumentParser(description="Load or reset the DocIntel Lab demo corpus.")
    parser.add_argument("--reset", action="store_true", help="Wipe demo data and rebuild the four corpus documents.")
    args = parser.parse_args()

    import django

    django.setup()
    if args.reset:
        reset()
    else:
        ingest_corpus()


def ingest_corpus() -> None:
    from scripts.make_synthetic_docs import build_all

    build_all()
    paths = sorted(path for path in (ROOT / "data" / "corpus").rglob("*") if path.suffix.lower() in SUFFIXES)
    if not paths:
        raise SystemExit(f"No documents under {ROOT / 'data' / 'corpus'}")
    _ingest(paths, strict=False)


def reset() -> None:
    from scripts.make_synthetic_docs import build_all

    from core.embeddings import describe as describe_embeddings
    from core.llm import describe_backend
    from core.models import EvalRun

    llm = describe_backend()
    embeddings = describe_embeddings()
    if llm["provider"] == "mock" or embeddings["provider"] == "mock":
        raise SystemExit(
            f"Refusing to reset. LLM is {llm['provider']} / {llm['model']}, "
            f"embeddings are {embeddings['provider']} / {embeddings['model']}. "
            "Set real providers in .env and restart the containers."
        )
    _step(f"LLM {llm['provider']} / {llm['model']} · embeddings {embeddings['provider']} / {embeddings['model']}")
    _step(f"Keeping {EvalRun.objects.count()} eval runs. Documents, rules, the graph and agent runs will be wiped.")
    _wipe()
    paths = build_all()
    if len(paths) != 4:
        raise SystemExit(f"Expected 4 corpus documents, got {len(paths)}.")
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Missing corpus file: {path}")
    _ingest(paths, strict=True)
    _extract_rules()
    _build_graph()
    _step("Demo reset complete. Open http://localhost:8010/")


def _wipe() -> None:
    from apps.agents.models import AgentRun
    from apps.ingest.models import Document
    from apps.kg.models import GraphBuild
    from apps.rules.models import RuleRun
    from core.graph_store import get_store

    runs = AgentRun.objects.count()
    AgentRun.objects.all().delete()
    _step(f"Deleted {runs} agent runs.")

    documents = list(Document.objects.all())
    for document in documents:
        if document.file:
            document.file.delete(save=False)
        document.delete()
    _step(f"Deleted {len(documents)} documents and their chunks, rules and extractions.")

    builds = GraphBuild.objects.count()
    GraphBuild.objects.all().delete()
    RuleRun.objects.all().delete()
    store = get_store()
    store.replace([], [])
    _step(f"Cleared the {store.name} graph and {builds} graph build rows.")


def _ingest(paths: list[Path], *, strict: bool) -> None:
    from apps.ingest.models import PipelineRun
    from apps.ingest.services.intake import ingest_path

    run_ids = []
    for path in paths:
        document, outcome = ingest_path(path)
        _step(f"{outcome:10} {path.name} -> document {document.pk} ({document.status})")
        if strict and outcome != "created":
            raise SystemExit(f"{path.name} came back as {outcome} after the wipe.")
        if outcome in {"created", "restarted"}:
            run_ids.append(document.runs.order_by("-id").first().pk)

    if not run_ids:
        _step("Nothing new to process. Same file hashes are already ingested.")
        _print_summary()
        return

    deadline = time.time() + INGEST_TIMEOUT_SECONDS
    while time.time() < deadline:
        runs = list(PipelineRun.objects.filter(id__in=run_ids))
        pending = [run for run in runs if run.status not in {PipelineRun.Status.DONE, PipelineRun.Status.FAILED}]
        if runs and not pending:
            break
        states = ", ".join(f"{run.document_id}:{run.status}/{run.current_stage or '-'}" for run in runs)
        _step(f"Waiting for the pipeline ({len(pending)} still running). {states}")
        time.sleep(5)
    else:
        raise SystemExit("Timed out waiting for the pipeline. Is the worker container running?")

    _print_summary()
    failed = [run for run in runs if run.status != PipelineRun.Status.DONE]
    if failed:
        for run in failed:
            stage = run.stages.filter(status="failed").first()
            detail = stage.error if stage else run.error
            print(f"FAILED {run.document.title} at {run.current_stage}: {detail}", flush=True)
        raise SystemExit(1)


def _extract_rules() -> None:
    from apps.rules.services.pipeline import run_rules

    _step("Extracting rules, merging duplicates and judging conflicts.")
    run = run_rules()
    extraction = run.stats["extraction"]
    conflicts = run.stats["conflicts"]
    totals = run.stats["totals"]
    _step(
        f"Rules: {extraction['ran']} chunks sent to the model, {extraction['skipped']} cached, "
        f"{extraction['failed']} failed validation."
    )
    _step(
        f"Extracted {totals['extracted']} · dropped {totals['dropped']} · merged {totals['merged']} · "
        f"conflicts {totals['conflicts']} · pending {totals['pending']} · retired {totals['retired']}"
    )
    _step(
        f"Conflict candidates {conflicts['candidates']}: {conflicts['judged']} judged, "
        f"{conflicts['cached']} cached, {conflicts['failed']} failed."
    )
    if extraction["failed"] or conflicts["failed"]:
        raise SystemExit("Rule extraction or conflict judging failed validation.")
    if totals["extracted"] == 0:
        raise SystemExit("No rules were extracted.")


def _build_graph() -> None:
    from apps.kg.services.build import build_graph

    _step("Extracting the graph and writing it.")
    build = build_graph(extract=True)
    _step(
        f"{build.backend}: {build.node_count} nodes, {build.edge_count} edges, "
        f"{build.merged_count} merged, {build.dropped_count} dropped, "
        f"from {build.chunk_count} chunks (build {build.id})"
    )
    for entry in build.merge_log:
        if "rejected_alias" in entry:
            _step(f"not merged into {entry['name']}: {entry['rejected_alias']} ({entry['reason']})")
    if build.node_count == 0:
        raise SystemExit("Graph build produced no nodes.")


def _print_summary() -> None:
    from django.db.models import Count

    from apps.ingest.models import Document

    print("", flush=True)
    print(f"{'document':48} {'type':6} {'status':8} {'chunks':6} {'pages':5}", flush=True)
    for document in Document.objects.annotate(chunk_count=Count("chunks")).order_by("id"):
        print(
            f"{document.title[:48]:48} {document.file_type:6} {document.status:8} "
            f"{document.chunk_count:<6} {document.page_count or 0:<5}",
            flush=True,
        )


def _step(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


if __name__ == "__main__":
    main()
