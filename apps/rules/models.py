from django.db import models

from apps.ingest.models import Chunk, Document


class RuleExtraction(models.Model):
    """The validated model output for one chunk and prompt version. An unchanged chunk is not re-run."""

    chunk = models.ForeignKey(Chunk, related_name="rule_extractions", on_delete=models.CASCADE)
    prompt_version = models.CharField(max_length=10)
    text_hash = models.CharField(max_length=64)
    llm_provider = models.CharField(max_length=20)
    llm_model = models.CharField(max_length=100, blank=True)
    raw_llm_output = models.JSONField(default=dict)
    kept_count = models.PositiveIntegerField(default=0)
    dropped = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["chunk_id"]
        constraints = [
            models.UniqueConstraint(fields=["chunk", "prompt_version"], name="uniq_rule_extraction"),
        ]

    def __str__(self) -> str:
        return f"rules {self.chunk_id} ({self.prompt_version})"


class Rule(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"
        MERGED = "merged", "Merged"
        RETIRED = "retired", "Retired"

    key = models.CharField(max_length=10, unique=True)
    document = models.ForeignKey(Document, related_name="rules", on_delete=models.CASCADE)
    chunk = models.ForeignKey(Chunk, related_name="rules", on_delete=models.CASCADE)
    page = models.PositiveIntegerField(null=True, blank=True)
    section = models.CharField(max_length=600, blank=True)
    condition = models.TextField()
    action = models.TextField()
    exception = models.TextField(blank=True)
    source_quote = models.TextField()
    confidence = models.FloatField(default=0.0)
    # Normalized condition + action as extracted. A re-extraction with the same fingerprint keeps the key.
    fingerprint = models.CharField(max_length=64, db_index=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    prompt_version = models.CharField(max_length=10)
    llm_provider = models.CharField(max_length=20)
    llm_model = models.CharField(max_length=100, blank=True)
    raw_llm_output = models.JSONField(default=dict)
    merged_into = models.ForeignKey("self", null=True, blank=True, related_name="duplicates", on_delete=models.SET_NULL)
    merge_reason = models.CharField(max_length=300, blank=True)
    edited = models.BooleanField(default=False)
    original = models.JSONField(default=dict, blank=True)
    reviewer_note = models.TextField(blank=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["key"]

    def __str__(self) -> str:
        return self.key

    def citation(self) -> str:
        parts = [self.document.title]
        if self.page:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(f"§ {self.section}")
        return ", ".join(parts)


class RuleRun(models.Model):
    """One extract -> merge -> conflict pass: counts, the merge log, and which models ran."""

    prompt_version = models.CharField(max_length=10)
    llm_provider = models.CharField(max_length=20)
    llm_model = models.CharField(max_length=100, blank=True)
    embeddings_provider = models.CharField(max_length=20, blank=True)
    embeddings_model = models.CharField(max_length=100, blank=True)
    forced = models.BooleanField(default=False)
    stats = models.JSONField(default=dict)
    merge_log = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"rules run {self.created_at:%Y-%m-%d %H:%M} ({self.prompt_version})"


class RuleConflict(models.Model):
    """The judge's verdict on one candidate pair. Only rows with is_conflict=True are conflicts.

    Non-conflicts are kept so an unchanged pair is not sent to the judge again; `signature`
    changes when either rule's text is edited.
    """

    class Severity(models.TextChoices):
        LOW = "low", "Low"
        MEDIUM = "medium", "Medium"
        HIGH = "high", "High"

    rule_a = models.ForeignKey(Rule, related_name="conflicts_as_a", on_delete=models.CASCADE)
    rule_b = models.ForeignKey(Rule, related_name="conflicts_as_b", on_delete=models.CASCADE)
    is_conflict = models.BooleanField(default=False)
    signature = models.CharField(max_length=64)
    kind = models.CharField(max_length=20)
    severity = models.CharField(max_length=10, choices=Severity.choices)
    reason = models.TextField()
    llm_provider = models.CharField(max_length=20)
    llm_model = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["rule_a", "rule_b"], name="uniq_rule_conflict"),
        ]

    def __str__(self) -> str:
        return f"{self.rule_a.key} vs {self.rule_b.key} ({self.severity})"
