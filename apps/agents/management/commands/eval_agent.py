from django.core.management.base import BaseCommand

from apps.agents.services import run_eval, table


class Command(BaseCommand):
    help = "Run evals/datasets/agent_cases.yaml through the compliance graph and score each case."

    def add_arguments(self, parser):
        parser.add_argument("--case", action="append", default=None, help="Only this case id (repeatable).")

    def handle(self, *args, **options):
        run = run_eval(only=options["case"])
        metrics = run.metrics
        self.stdout.write(f"LLM {run.llm_provider} / {run.llm_model} · embeddings {run.embeddings_provider} / "
                          f"{run.embeddings_model} · retrieval {run.config['retrieval_mode']}")
        for line in table(metrics["rows"]):
            self.stdout.write(line)
        self.stdout.write(
            f"intent {metrics['intent_correct']}/{metrics['cases']} · answer {metrics['answer_correct']}/{metrics['cases']}"
            f" · citations verified {metrics['citations_verified']}/{metrics['citation_cases']}"
            f" · conflict check {metrics['conflict_surfaced']}/{metrics['conflict_cases']} · loops {metrics['loops_total']}"
        )
        self.stdout.write(f"Answers and failures are in EvalRun {run.id} and evals/results/.")
