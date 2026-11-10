from django.db import models
from pgvector.django import HnswIndex, VectorField

from core.embeddings import EMBEDDING_DIM


class Document(models.Model):
    """One uploaded file. `sha256` makes a second upload of the same bytes a no-op."""

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    title = models.CharField(max_length=255)
    original_name = models.CharField(max_length=255)
    file = models.FileField(upload_to="uploads/%Y/%m/")
    file_type = models.CharField(max_length=10, blank=True)
    sha256 = models.CharField(max_length=64, unique=True)
    size_bytes = models.PositiveIntegerField()
    page_count = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title


class PipelineRun(models.Model):
    """One pass of a document through the stage tasks."""

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    document = models.ForeignKey(Document, related_name="runs", on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED)
    current_stage = models.CharField(max_length=20, blank=True)
    error = models.TextField(blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-id"]

    def __str__(self) -> str:
        return f"run {self.pk} of {self.document_id} ({self.status})"


class PipelineStage(models.Model):
    """Timing, attempts and outcome of one stage inside a run."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        RETRYING = "retrying", "Retrying"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    run = models.ForeignKey(PipelineRun, related_name="stages", on_delete=models.CASCADE)
    name = models.CharField(max_length=20)
    position = models.PositiveSmallIntegerField()
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    attempts = models.PositiveSmallIntegerField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    duration_ms = models.PositiveIntegerField(default=0)
    error = models.TextField(blank=True)
    summary = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["position"]
        constraints = [
            models.UniqueConstraint(fields=["run", "name"], name="uniq_stage_per_run"),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.status})"


class Chunk(models.Model):
    """A structure-aware piece of a document, with the vector used for later retrieval."""

    document = models.ForeignKey(Document, related_name="chunks", on_delete=models.CASCADE)
    ordinal = models.PositiveIntegerField()
    text = models.TextField()
    text_hash = models.CharField(max_length=64)
    page_start = models.PositiveIntegerField(null=True, blank=True)
    page_end = models.PositiveIntegerField(null=True, blank=True)
    section_title = models.CharField(max_length=300, blank=True)
    section_path = models.CharField(max_length=600, blank=True)
    heading_level = models.PositiveSmallIntegerField(null=True, blank=True)
    token_count = models.PositiveIntegerField(default=0)
    metadata = models.JSONField(default=dict, blank=True)
    embedding = VectorField(dimensions=EMBEDDING_DIM, null=True, blank=True)

    class Meta:
        ordering = ["ordinal"]
        constraints = [
            models.UniqueConstraint(fields=["document", "ordinal"], name="uniq_chunk_ordinal"),
        ]
        indexes = [
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]

    def __str__(self) -> str:
        return f"{self.document_id}#{self.ordinal}"
