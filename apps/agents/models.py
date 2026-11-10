from django.db import models


class AgentRun(models.Model):
    """One question through the compliance graph. `thread_id` is the LangGraph checkpointer thread."""

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        AWAITING_APPROVAL = "awaiting_approval", "Awaiting approval"
        DONE = "done", "Approved"
        REJECTED = "rejected", "Rejected"
        REFUSED = "refused", "Out of scope"
        FAILED = "failed", "Failed"

    question = models.TextField()
    thread_id = models.CharField(max_length=64, unique=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED)
    source = models.CharField(max_length=10, default="ui")  # ui | eval | cli
    intent = models.CharField(max_length=30, blank=True)
    loops = models.PositiveSmallIntegerField(default=0)
    verified = models.BooleanField(null=True)
    warning = models.TextField(blank=True)
    draft = models.JSONField(default=dict, blank=True)
    # What the reviewer needs next to the draft: conflicts of applicable rules and the citation checks.
    review = models.JSONField(default=dict, blank=True)
    final = models.JSONField(default=dict, blank=True)
    approved = models.BooleanField(null=True)
    approval_note = models.TextField(blank=True)
    llm_provider = models.CharField(max_length=20)
    llm_model = models.CharField(max_length=100, blank=True)
    retrieval_mode = models.CharField(max_length=20, blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"agent run {self.id} ({self.status})"

    @property
    def active(self) -> bool:
        return self.status in (self.Status.QUEUED, self.Status.RUNNING)


class AgentStep(models.Model):
    """One node execution: what it read, what it wrote, the tools and LLM calls it made, and how long it took."""

    class Status(models.TextChoices):
        DONE = "done", "Done"
        WAITING = "waiting", "Waiting"
        FAILED = "failed", "Failed"

    run = models.ForeignKey(AgentRun, related_name="steps", on_delete=models.CASCADE)
    order = models.PositiveIntegerField()
    node = models.CharField(max_length=40)
    loop = models.PositiveSmallIntegerField(default=0)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.DONE)
    input = models.JSONField(default=dict, blank=True)
    output = models.JSONField(default=dict, blank=True)
    tool_calls = models.JSONField(default=list, blank=True)
    llm_calls = models.JSONField(default=list, blank=True)
    duration_ms = models.PositiveIntegerField(default=0)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["run_id", "order"]

    def __str__(self) -> str:
        return f"{self.run_id}.{self.order} {self.node}"
