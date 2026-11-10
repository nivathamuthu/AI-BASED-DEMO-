from django.core.management.base import BaseCommand, CommandError

from apps.kg.models import GraphBuild
from apps.kg.services.eval import run_eval
from apps.rag.services.bm25 import clear_cache
from core.graph_store import GraphStoreError


class Command(BaseCommand):
    help = "Compare plain hybrid RAG with GraphRAG on evals/datasets/kg_questions.yaml."

    def handle(self, *args, **options):
        if not GraphBuild.objects.exists():
            raise CommandError("No graph yet. Run: python manage.py build_graph --extract")
        clear_cache()
        try:
            run = run_eval()
        except GraphStoreError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(run.report_md)
        self.stdout.write(f"Stored EvalRun {run.id}")
