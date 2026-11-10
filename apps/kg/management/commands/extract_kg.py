from django.core.management.base import BaseCommand, CommandError

from apps.ingest.models import Chunk
from apps.kg.services.extract import extract_chunks


class Command(BaseCommand):
    help = "Extract entities and relations per chunk into KGExtraction. Cached chunks are skipped."

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="Re-extract chunks that already have a row.")
        parser.add_argument("--prompt-version", default=None, help="v1 or v2 (default KG_PROMPT_VERSION).")
        parser.add_argument("--document", type=int, default=None, help="Only this document id.")

    def handle(self, *args, **options):
        chunks = Chunk.objects.order_by("id")
        if options["document"]:
            chunks = chunks.filter(document_id=options["document"])
        if not chunks.exists():
            raise CommandError("No chunks. Ingest documents first: .\\make.ps1 demo")
        stats = extract_chunks(chunks, version=options["prompt_version"], force=options["force"])
        self.stdout.write(
            f"extracted {stats['extracted']}, skipped {stats['skipped']} (cached), failed {stats['failed']}, "
            f"kept {stats['relations']} relations, dropped {stats['dropped']}"
        )
