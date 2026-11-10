"""One function per pipeline stage. Each returns a JSON-serializable summary.

Stages talk through files under MEDIA_ROOT/pipeline/<document id>/, because the
Celery tasks that call them are separate processes.
"""
import hashlib
import re
from pathlib import Path

import numpy as np
from django.conf import settings

from apps.ingest.models import Chunk, Document
from apps.ingest.services import artifacts
from apps.ingest.services.chunk import ChunkDraft, chunk_blocks
from apps.ingest.services.clean import clean_blocks
from apps.ingest.services.detect import detect_file_type
from apps.ingest.services.parse import parse_document
from core.embeddings import EMBEDDING_DIM, embed_texts, model_name

# Identifiers worth surfacing in the UI. These run on document text, not on LLM output.
_PATTERNS = {
    "ad": re.compile(r"\bAD\s+\d{4}-\d{2}-\d{2}\b", re.I),
    "part": re.compile(r"\b(?:P/?N|PN)[-\s]?[A-Z0-9]+(?:-[A-Z0-9]+)+\b", re.I),
    "part_number": re.compile(r"\b\d{2}-\d{4,6}-\d{1,4}\b"),
}


def run_detect(document: Document) -> dict:
    file_type = detect_file_type(Path(document.file.path))
    document.file_type = file_type
    document.save(update_fields=["file_type", "updated_at"])
    return {"file_type": file_type, "bytes": document.size_bytes}


def run_parse(document: Document) -> dict:
    blocks = parse_document(Path(document.file.path), document.file_type)
    artifacts.save_blocks(document.id, "blocks", blocks)
    pages = [block.page for block in blocks if block.page]
    document.page_count = max(pages) if pages else None
    document.save(update_fields=["page_count", "updated_at"])
    return {
        "blocks": len(blocks),
        "headings": sum(1 for block in blocks if block.is_heading),
        "pages": document.page_count,
    }


def run_clean(document: Document) -> dict:
    blocks = artifacts.load_blocks(document.id, "blocks")
    cleaned = clean_blocks(blocks)
    artifacts.save_blocks(document.id, "cleaned", cleaned)
    return {"blocks_in": len(blocks), "blocks_out": len(cleaned), "removed": len(blocks) - len(cleaned)}


def run_chunk(document: Document) -> dict:
    blocks = artifacts.load_blocks(document.id, "cleaned")
    drafts = chunk_blocks(
        blocks,
        max_tokens=settings.INGEST_CHUNK_MAX_TOKENS,
        min_tokens=settings.INGEST_CHUNK_MIN_TOKENS,
    )
    artifacts.save_json(document.id, "chunks", [draft.to_dict() for draft in drafts])
    return {
        "chunks": len(drafts),
        "tokens": sum(draft.token_count for draft in drafts),
        "max_tokens": settings.INGEST_CHUNK_MAX_TOKENS,
    }


def run_enrich(document: Document) -> dict:
    drafts = [ChunkDraft.from_dict(item) for item in artifacts.load_json(document.id, "chunks")]
    found: list[str] = []
    enriched = []
    for draft in drafts:
        metadata = _metadata(draft, document)
        enriched.append({**draft.to_dict(), "metadata": metadata})
        found.extend(item["value"] for item in metadata["identifiers"])
    artifacts.save_json(document.id, "enriched", enriched)
    unique = sorted(set(found))
    return {"chunks": len(enriched), "identifiers": unique}


def run_embed(document: Document) -> dict:
    enriched = artifacts.load_json(document.id, "enriched")
    vectors = embed_texts([item["text"] for item in enriched])
    if vectors.shape != (len(enriched), EMBEDDING_DIM):
        raise ValueError(f"Expected embeddings of shape {(len(enriched), EMBEDDING_DIM)}, got {vectors.shape}")
    np.save(artifacts.directory(document.id) / "embeddings.npy", vectors)
    return {"chunks": len(enriched), "dimensions": EMBEDDING_DIM, "model": model_name()}


def run_index(document: Document) -> dict:
    enriched = artifacts.load_json(document.id, "enriched")
    vectors = np.load(artifacts.directory(document.id) / "embeddings.npy")
    if len(vectors) != len(enriched):
        raise ValueError(f"{len(vectors)} embeddings for {len(enriched)} chunks")
    document.chunks.all().delete()
    rows = []
    for item, vector in zip(enriched, vectors):
        metadata = item["metadata"]
        metadata["embedding_model"] = model_name()
        metadata["embedding_provider"] = settings.EMBEDDINGS_PROVIDER
        rows.append(
            Chunk(
                document=document,
                ordinal=item["ordinal"],
                text=item["text"],
                text_hash=hashlib.sha256(item["text"].encode()).hexdigest(),
                page_start=item["page_start"],
                page_end=item["page_end"],
                section_title=item["section_title"][:300],
                section_path=item["section_path"][:600],
                heading_level=item["heading_level"],
                token_count=item["token_count"],
                metadata=metadata,
                embedding=vector.tolist(),
            )
        )
    Chunk.objects.bulk_create(rows)
    return {"chunks": len(rows), "dimensions": EMBEDDING_DIM}


STAGE_FUNCTIONS = {
    "detect": run_detect,
    "parse": run_parse,
    "clean": run_clean,
    "chunk": run_chunk,
    "enrich": run_enrich,
    "embed": run_embed,
    "index": run_index,
}


def _metadata(draft: ChunkDraft, document: Document) -> dict:
    identifiers = []
    seen = set()
    for kind, pattern in _PATTERNS.items():
        for match in pattern.findall(draft.text):
            value = " ".join(match.split())
            label = "part" if kind == "part_number" else kind
            key = (label, value.upper())
            if key in seen:
                continue
            seen.add(key)
            identifiers.append({"kind": label, "value": value})
    return {
        "file_type": document.file_type,
        "source_name": document.original_name,
        "token_count": draft.token_count,
        "char_count": len(draft.text),
        "identifiers": identifiers,
    }
