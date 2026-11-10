"""Rebuild every chunk vector with the current EMBEDDINGS_PROVIDER."""
from django.core.management.base import BaseCommand, CommandError

from apps.ingest.models import Chunk
from apps.rag.services.bm25 import clear_cache
from core.embeddings import EMBEDDING_DIM, describe, embed_texts


class Command(BaseCommand):
    help = "Re-embed all chunks. Required after changing EMBEDDINGS_PROVIDER."

    def handle(self, *args, **options):
        chunks = list(Chunk.objects.order_by("id"))
        if not chunks:
            raise CommandError("No chunks yet. Run .\\make.ps1 demo first.")
        info = describe()
        self.stdout.write(f"Embedding {len(chunks)} chunks with {info['provider']} / {info['model']} ({info['dimensions']}-dim)")
        vectors = embed_texts([chunk.text for chunk in chunks])
        if vectors.shape[1] != EMBEDDING_DIM:
            raise CommandError(f"Expected {EMBEDDING_DIM} dimensions, got {vectors.shape[1]}")
        for chunk, vector in zip(chunks, vectors):
            metadata = dict(chunk.metadata or {})
            metadata["embedding_provider"] = info["provider"]
            metadata["embedding_model"] = info["model"]
            chunk.embedding = vector.tolist()
            chunk.metadata = metadata
            chunk.save(update_fields=["embedding", "metadata"])
        clear_cache()
        self.stdout.write(self.style.SUCCESS(f"Re-embedded {len(chunks)} chunks."))
