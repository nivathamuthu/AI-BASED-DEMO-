from django.core.management.base import BaseCommand

from apps.rules.services.pipeline import run_rules


class Command(BaseCommand):
    help = "Extract rules from every chunk, drop unquoted rules, merge duplicates, judge conflicts."

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="Re-run the model on unchanged chunks.")
        parser.add_argument("--prompt-version", default=None, help="v1 or v2 (default RULES_PROMPT_VERSION).")
        parser.add_argument("--document", type=int, action="append", help="Only this document id. Repeatable.")

    def handle(self, *args, **options):
        run = run_rules(document_ids=options["document"], version=options["prompt_version"], force=options["force"])
        extraction = run.stats["extraction"]
        conflicts = run.stats["conflicts"]
        totals = run.stats["totals"]
        self.stdout.write(f"LLM {run.llm_provider} / {run.llm_model} · embeddings {run.embeddings_provider} / "
                          f"{run.embeddings_model} · prompt {run.prompt_version}")
        if run.llm_provider == "mock":
            self.stdout.write(self.style.WARNING("LLM_PROVIDER=mock: rules come from the offline stand-in, not a model."))
        self.stdout.write(
            f"chunks {extraction['chunks']}: {extraction['ran']} sent to the model, {extraction['skipped']} unchanged "
            f"(cached), {extraction['failed']} failed validation"
        )
        self.stdout.write(
            f"rules extracted {totals['extracted']} · dropped (quote not found) {totals['dropped']} · "
            f"merged {totals['merged']} · conflicts {totals['conflicts']} · approved {totals['approved']} · "
            f"pending {totals['pending']} · retired {totals['retired']}"
        )
        self.stdout.write(
            f"conflict candidates {conflicts['candidates']}: {conflicts['judged']} judged, "
            f"{conflicts['cached']} cached, {conflicts['failed']} failed"
        )
        for entry in run.merge_log[:10]:
            self.stdout.write(f"  merged {entry['rule']} into {entry['into']}: {entry['reason']}")
