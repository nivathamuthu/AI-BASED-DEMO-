from django.core.management.base import BaseCommand

from apps.rules.services.eval import run_eval, table


class Command(BaseCommand):
    help = "Precision and recall of extract_rules v1 and v2 against evals/datasets/rules_gold.yaml."

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="Re-run the model instead of using cached output.")

    def handle(self, *args, **options):
        run = run_eval(force=options["force"])
        self.stdout.write(f"LLM {run.llm_provider} / {run.llm_model} · embeddings {run.embeddings_provider} / "
                          f"{run.embeddings_model}")
        if run.config.get("missing_documents"):
            self.stdout.write(self.style.WARNING(f"Not ingested: {', '.join(run.config['missing_documents'])}"))
        for line in table(run.metrics):
            self.stdout.write(line)
        self.stdout.write(f"Full report saved as EvalRun {run.id} and in evals/results/.")
