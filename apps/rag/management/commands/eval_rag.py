"""15 questions. Each `expect` string occurs in the synthetic maintenance corpus."""
from django.core.management.base import BaseCommand, CommandError

from apps.ingest.models import Chunk
from apps.rag.services.bm25 import clear_cache
from apps.rag.services.eval import run_eval


class Command(BaseCommand):
    help = "Report hit@5 for vector, BM25, hybrid and hybrid+rerank."

    def handle(self, *args, **options):
        if not Chunk.objects.exists():
            raise CommandError("No chunks yet. Run .\\make.ps1 demo first.")
        clear_cache()
        run = run_eval()
        self.stdout.write(run.report_md)
        self.stdout.write(self.style.SUCCESS(f"Stored EvalRun {run.pk}"))
