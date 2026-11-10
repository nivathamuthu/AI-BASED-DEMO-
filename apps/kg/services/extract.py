"""Per-chunk entity and relation extraction, cached in KGExtraction."""
import logging

from django.conf import settings

from apps.ingest.models import Chunk
from apps.kg.models import KGExtraction
from apps.kg.schemas import KGOutput
from apps.kg.services.normalize import entity_key
from core.llm import LLMValidationError, describe_backend, structured_call
from core.prompts import render_prompt

logger = logging.getLogger(__name__)


def _flat(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .\"'")


def filter_relations(output: KGOutput, chunk_text: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Keep relations whose endpoints are extracted entities and whose evidence is in the chunk."""
    entities = [entity.model_dump() for entity in output.entities]
    known = set()
    for entity in output.entities:
        known.add(entity_key(entity.name))
        known.update(entity_key(alias) for alias in entity.aliases)
    haystack = _flat(chunk_text)
    kept, dropped = [], []
    for relation in output.relations:
        row = relation.model_dump()
        missing = [end for end in (relation.source, relation.target) if entity_key(end) not in known]
        if missing:
            dropped.append({**row, "reason": f"endpoint not in entity list: {', '.join(missing)}"})
        elif _flat(relation.evidence_quote) not in haystack:
            dropped.append({**row, "reason": "evidence_quote is not in the chunk"})
        else:
            kept.append(row)
    return entities, kept, dropped


def extract_chunk(chunk: Chunk, version: str | None = None, force: bool = False) -> tuple[KGExtraction, bool]:
    """Return (row, ran). Skips the LLM when this chunk already has an extraction with `version`."""
    version = version or settings.KG_PROMPT_VERSION
    existing = KGExtraction.objects.filter(chunk=chunk).first()
    backend = describe_backend()
    same_model = existing and (existing.llm_provider, existing.llm_model) == (backend["provider"], backend["model"])
    if existing and same_model and existing.prompt_version == version and not force:
        return existing, False
    prompt = render_prompt(
        "extract_kg", version, {"chunk": chunk.text, "domain": "aviation maintenance"}, KGOutput
    )
    output = structured_call(
        KGOutput, prompt, task="kg.extract", prompt_name="extract_kg", prompt_version=version
    )
    entities, relations, dropped = filter_relations(output, chunk.text)
    if dropped:
        logger.info(
            "kg chunk %s: dropped %d of %d relations (%s)",
            chunk.id, len(dropped), len(dropped) + len(relations),
            "; ".join(sorted({row["reason"] for row in dropped}))[:300],
        )
    backend = describe_backend()
    row, _created = KGExtraction.objects.update_or_create(
        chunk=chunk,
        defaults={
            "prompt_version": version,
            "llm_provider": backend["provider"],
            "llm_model": backend["model"],
            "entities": entities,
            "relations": relations,
            "dropped": dropped,
        },
    )
    return row, True


def extract_chunks(chunks, version: str | None = None, force: bool = False) -> dict:
    """Extract every chunk in `chunks`. One failing chunk is logged and skipped."""
    stats = {"extracted": 0, "skipped": 0, "failed": 0, "dropped": 0, "relations": 0}
    for chunk in chunks:
        try:
            row, ran = extract_chunk(chunk, version=version, force=force)
        except LLMValidationError as exc:
            logger.warning("kg chunk %s failed validation: %s", chunk.id, exc)
            stats["failed"] += 1
            continue
        stats["extracted" if ran else "skipped"] += 1
        if ran:
            stats["dropped"] += len(row.dropped)
            stats["relations"] += len(row.relations)
    return stats
