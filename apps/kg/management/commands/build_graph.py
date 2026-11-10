from django.core.management.base import BaseCommand, CommandError

from apps.kg.models import KGExtraction
from apps.kg.services.build import build_graph
from core.graph_store import GraphStoreError


class Command(BaseCommand):
    help = "Rebuild the graph store from KGExtraction rows. No LLM calls unless --extract."

    def add_arguments(self, parser):
        parser.add_argument("--extract", action="store_true", help="Extract chunks that have no row first.")
        parser.add_argument("--force", action="store_true", help="With --extract, re-extract every chunk.")
        parser.add_argument("--prompt-version", default=None, help="Prompt version for --extract, v1 or v2.")

    def handle(self, *args, **options):
        if not options["extract"] and not KGExtraction.objects.exists():
            raise CommandError("No KGExtraction rows. Run with --extract, or run extract_kg first.")
        try:
            build = build_graph(extract=options["extract"], force=options["force"], version=options["prompt_version"])
        except GraphStoreError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(
            f"{build.backend}: {build.node_count} nodes, {build.edge_count} edges, "
            f"{build.merged_count} merged duplicates, {build.dropped_count} dropped relations, "
            f"from {build.chunk_count} chunks (build {build.id})"
        )
        for entry in build.merge_log[:15]:
            if "merged" in entry:
                forms = ", ".join(f"{item['form']} ({item['reason']})" for item in entry["merged"])
                self.stdout.write(f"  merged into {entry['name']}: {forms}")
        for entry in build.merge_log:
            if "rejected_alias" in entry:
                self.stdout.write(f"  not merged into {entry['name']}: {entry['rejected_alias']} ({entry['reason']})")
