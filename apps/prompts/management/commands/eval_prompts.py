from django.core.management.base import BaseCommand

from apps.prompts.services.eval import run_eval


class Command(BaseCommand):
    help = "Compare prompt v1 and v2 on evals/datasets/prompt_samples.yaml."

    def handle(self, *args, **options):
        run = run_eval()
        self.stdout.write(run.report_md)
        self.stdout.write(self.style.SUCCESS(f"Stored EvalRun {run.pk}"))
