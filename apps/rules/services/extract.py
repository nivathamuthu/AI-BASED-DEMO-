"""Per-chunk rule extraction, cached in RuleExtraction, synced into Rule rows with stable keys."""
import logging

from django.conf import settings
from django.db import transaction

from apps.ingest.models import Chunk
from apps.rules.models import Rule, RuleExtraction
from apps.rules.schemas import RuleSet
from apps.rules.services.text import fingerprint, normalize_quote, quote_in_chunk
from core.llm import describe_backend, structured_call
from core.prompts import render_prompt

logger = logging.getLogger(__name__)

PROMPT_TASK = "extract_rules"


def verify_quotes(rules: list[dict], chunk_text: str) -> tuple[list[dict], list[dict]]:
    """Split extracted rules into (kept, dropped). A rule is kept only if its quote is in the chunk."""
    kept, dropped = [], []
    seen = set()
    for rule in rules:
        if not quote_in_chunk(rule.get("source_quote", ""), chunk_text):
            dropped.append({**rule, "reason": "source_quote not found in chunk"})
            continue
        key = fingerprint(rule["condition"], rule["action"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(rule)
    return kept, dropped


def run_extraction(chunk: Chunk, version: str, force: bool = False) -> tuple[RuleExtraction, bool]:
    """Return (row, ran). The model is called only when the chunk text or prompt version is new, or forced."""
    backend = describe_backend()
    existing = RuleExtraction.objects.filter(chunk=chunk, prompt_version=version).first()
    same_model = existing and (existing.llm_provider, existing.llm_model) == (backend["provider"], backend["model"])
    if existing and same_model and existing.text_hash == chunk.text_hash and not force:
        return existing, False
    prompt = render_prompt(PROMPT_TASK, version, {"chunk": chunk.text, "domain": "aviation maintenance"}, RuleSet)
    output = structured_call(RuleSet, prompt, task="rules.extract", prompt_name=PROMPT_TASK, prompt_version=version)
    raw = output.model_dump()
    kept, dropped = verify_quotes(raw["rules"], chunk.text)
    if dropped:
        logger.info("rules chunk %s (%s): dropped %d of %d rules, quote not in chunk",
                    chunk.id, version, len(dropped), len(raw["rules"]))
    row, _created = RuleExtraction.objects.update_or_create(
        chunk=chunk,
        prompt_version=version,
        defaults={
            "text_hash": chunk.text_hash,
            "llm_provider": backend["provider"],
            "llm_model": backend["model"],
            "raw_llm_output": raw,
            "kept_count": len(kept),
            "dropped": dropped,
        },
    )
    return row, True


def kept_rules(extraction: RuleExtraction) -> list[dict]:
    kept, _dropped = verify_quotes(extraction.raw_llm_output.get("rules", []), extraction.chunk.text)
    return kept


def _next_key_number() -> int:
    last = Rule.objects.order_by("-key").values_list("key", flat=True).first()
    return int(last.split("-")[1]) + 1 if last else 1


@transaction.atomic
def sync_rules(extraction: RuleExtraction) -> dict:
    """Make the chunk's Rule rows match the extraction. Matching rules keep their key and review state.

    A pending rule the new extraction no longer contains is retired, not deleted. Reviewed rules
    (approved, rejected) are never changed by extraction.
    """
    chunk = extraction.chunk
    existing = list(Rule.objects.select_for_update().filter(chunk=chunk))
    by_fingerprint = {rule.fingerprint: rule for rule in existing}
    by_quote = {normalize_quote(rule.source_quote): rule for rule in existing}
    reviewed = {Rule.Status.APPROVED, Rule.Status.REJECTED}
    used: set[int] = set()
    stats = {"created": 0, "updated": 0, "retired": 0}
    next_number = _next_key_number()
    for item in kept_rules(extraction):
        rule_fingerprint = fingerprint(item["condition"], item["action"])
        rule = by_fingerprint.get(rule_fingerprint) or by_quote.get(normalize_quote(item["source_quote"]))
        if rule is not None and rule.id in used:
            rule = None
        if rule is None:
            rule = Rule(key=f"R-{next_number:04d}", document_id=chunk.document_id, chunk=chunk)
            next_number += 1
            stats["created"] += 1
        else:
            stats["updated"] += 1
        if rule.status not in reviewed and not rule.edited:
            rule.condition = item["condition"]
            rule.action = item["action"]
            rule.exception = item.get("exception", "")
            rule.source_quote = item["source_quote"]
            rule.confidence = item["confidence"]
            rule.fingerprint = rule_fingerprint
        if rule.status == Rule.Status.RETIRED:
            rule.status = Rule.Status.PENDING
        rule.page = chunk.page_start
        rule.section = chunk.section_path or chunk.section_title
        rule.prompt_version = extraction.prompt_version
        rule.llm_provider = extraction.llm_provider
        rule.llm_model = extraction.llm_model
        rule.raw_llm_output = {"prompt_version": extraction.prompt_version, "rule": item}
        rule.save()
        used.add(rule.id)
    for rule in existing:
        if rule.id not in used and rule.status in (Rule.Status.PENDING, Rule.Status.MERGED):
            rule.status = Rule.Status.RETIRED
            rule.merged_into = None
            rule.save(update_fields=["status", "merged_into", "updated_at"])
            stats["retired"] += 1
    return stats


def extract_chunks(chunks, version: str | None = None, force: bool = False) -> dict:
    """Extract and sync every chunk. One chunk failing validation is logged and skipped."""
    from core.llm import LLMValidationError

    version = version or settings.RULES_PROMPT_VERSION
    stats = {"chunks": 0, "ran": 0, "skipped": 0, "failed": 0, "kept": 0, "dropped": 0,
             "created": 0, "updated": 0, "retired": 0}
    for chunk in chunks:
        stats["chunks"] += 1
        try:
            extraction, ran = run_extraction(chunk, version, force=force)
        except LLMValidationError as exc:
            logger.warning("rules chunk %s failed validation: %s", chunk.id, exc)
            stats["failed"] += 1
            continue
        stats["ran" if ran else "skipped"] += 1
        stats["kept"] += extraction.kept_count
        stats["dropped"] += len(extraction.dropped)
        for name, value in sync_rules(extraction).items():
            stats[name] += value
    logger.info("rules extraction %s: %s", version, stats)
    return stats
