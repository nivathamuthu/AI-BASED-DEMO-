from django.db import models

from apps.ingest.models import Chunk


class KGExtraction(models.Model):
    """Validated entities and relations for one chunk, so the graph rebuilds without new LLM calls."""

    chunk = models.OneToOneField(Chunk, related_name="kg_extraction", on_delete=models.CASCADE)
    prompt_version = models.CharField(max_length=10)
    llm_provider = models.CharField(max_length=20)
    llm_model = models.CharField(max_length=100, blank=True)
    entities = models.JSONField(default=list)
    relations = models.JSONField(default=list)
    dropped = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["chunk_id"]

    def __str__(self) -> str:
        return f"kg {self.chunk_id} ({self.prompt_version})"


class GraphBuild(models.Model):
    """One rebuild of the graph store from KGExtraction rows."""

    backend = models.CharField(max_length=20)
    prompt_version = models.CharField(max_length=10, blank=True)
    node_count = models.PositiveIntegerField(default=0)
    edge_count = models.PositiveIntegerField(default=0)
    merged_count = models.PositiveIntegerField(default=0)
    dropped_count = models.PositiveIntegerField(default=0)
    chunk_count = models.PositiveIntegerField(default=0)
    merge_log = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"graph {self.backend} {self.node_count}n/{self.edge_count}e @ {self.created_at:%Y-%m-%d %H:%M}"
