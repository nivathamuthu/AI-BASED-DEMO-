from django.core.management.base import BaseCommand, CommandError

from apps.agents import graph
from apps.agents.models import AgentRun


class Command(BaseCommand):
    help = "Run one question through the compliance graph and print the trace. Stops at the approval step."

    def add_arguments(self, parser):
        parser.add_argument("question")
        parser.add_argument("--approve", action="store_true", help="Approve the paused run and print the final answer.")
        parser.add_argument("--reject", default=None, metavar="NOTE", help="Reject the paused run with this note.")

    def handle(self, *args, **options):
        run = graph.create_run(options["question"], source="cli")
        try:
            run = graph.execute(run)
            if run.status == AgentRun.Status.AWAITING_APPROVAL and (options["approve"] or options["reject"]):
                run = graph.resume(run, approved=options["reject"] is None, note=options["reject"] or "approved from CLI")
        except Exception as exc:
            raise CommandError(f"run {run.id} failed: {exc}") from exc
        self.stdout.write(f"run {run.id} · LLM {run.llm_provider} / {run.llm_model} · thread {run.thread_id}")
        for step in run.steps.all():
            tools = ", ".join(call["tool"] for call in step.tool_calls) or "-"
            llm = ", ".join(f"{call['schema']} x{call['attempts']}" for call in step.llm_calls) or "-"
            self.stdout.write(f"  {step.order:>2} {step.node:<17} loop {step.loop} {step.status:<7} "
                              f"{step.duration_ms:>6} ms  tools: {tools}  llm: {llm}")
            notes = step.output.get("notes", {})
            if notes:
                self.stdout.write(f"     {notes}")
        self.stdout.write(f"status {run.status} · intent {run.intent} · loops {run.loops} · verified {run.verified}")
        if run.warning:
            self.stdout.write(self.style.WARNING(run.warning))
        for conflict in run.review.get("conflicts", []):
            self.stdout.write(self.style.WARNING(f"conflict {conflict['rule']} vs {conflict['other']}: {conflict['reason']}"))
        body = run.final or run.draft
        self.stdout.write(f"answer: {body.get('answer', '')}")
        for citation in (run.final or {}).get("citations", []) or [
            {"n": i, "citation": f"S{c['source']}", "quote": c["quote"], "verified": None}
            for i, c in enumerate(run.draft.get("claims", []), start=1)
        ]:
            self.stdout.write(f"  [{citation['n']}] {citation['citation']} \"{citation['quote']}\" verified={citation['verified']}")
        if run.status == AgentRun.Status.AWAITING_APPROVAL:
            self.stdout.write(f"Paused for approval. Open http://localhost:8010/agents/run/{run.id}/")
