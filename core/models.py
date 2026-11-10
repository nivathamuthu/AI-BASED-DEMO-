from django.db import models


class LLMCall(models.Model):
    """One structured LLM call (all attempts included). Used for cost/latency/validity stats."""

    task = models.CharField(max_length=100)
    schema_name = models.CharField(max_length=100)
    prompt_name = models.CharField(max_length=100, blank=True)
    prompt_version = models.CharField(max_length=20, blank=True)
    provider = models.CharField(max_length=20)
    model = models.CharField(max_length=100, blank=True)
    attempts = models.PositiveSmallIntegerField(default=1)
    validation_ok = models.BooleanField(default=False)
    error = models.TextField(blank=True)
    latency_ms = models.PositiveIntegerField(default=0)
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        status = "ok" if self.validation_ok else "failed"
        return f"{self.task} [{self.provider}] {status} in {self.attempts} attempt(s)"


class EvalRun(models.Model):
    """One prompt or retrieval evaluation, including which models produced it."""

    kind = models.CharField(max_length=20)
    llm_provider = models.CharField(max_length=20)
    llm_model = models.CharField(max_length=100, blank=True)
    embeddings_provider = models.CharField(max_length=20, blank=True)
    embeddings_model = models.CharField(max_length=150, blank=True)
    config = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    report_md = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.kind} {self.llm_provider}/{self.embeddings_provider or '-'} @ {self.created_at:%Y-%m-%d %H:%M}"
