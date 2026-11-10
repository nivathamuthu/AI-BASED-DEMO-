"""Merge near-duplicate rules into one canonical rule, keeping merged_into links and a log."""
import logging

import numpy as np
from django.conf import settings
from django.db import transaction

from apps.rules.models import Rule
from apps.rules.services.text import content_words, identifiers, jaccard, thresholds
from core.embeddings import embed_texts

logger = logging.getLogger(__name__)

ACTIVE = (Rule.Status.PENDING, Rule.Status.APPROVED)


def rule_text(rule: Rule) -> str:
    return f"IF {rule.condition} THEN {rule.action}"


def _numbers(rule: Rule) -> set[str]:
    text = f"{rule.condition} {rule.action} {rule.exception}"
    return identifiers(text) | thresholds(text)


def duplicate_reason(left: Rule, right: Rule, cosine: float, threshold: float) -> str:
    """Why two rules are duplicates, or "" when they are not. Different numbers are never duplicates."""
    if _numbers(left) != _numbers(right):
        return ""
    if left.fingerprint == right.fingerprint:
        return "same normalized condition and action"
    overlap = jaccard(content_words(rule_text(left)), content_words(rule_text(right)))
    if cosine >= threshold and overlap >= 0.5:
        return f"cosine {cosine:.2f}, word overlap {overlap:.2f}, same numbers"
    return ""


def _canonical_order(rule: Rule) -> tuple:
    return (rule.status != Rule.Status.APPROVED, -rule.confidence, rule.key)


@transaction.atomic
def merge_duplicates(threshold: float | None = None) -> list[dict]:
    """Recompute automatic merges over active rules. Returns the merge log.

    Previous automatic merges are undone first, so a rule that was edited or rejected
    since the last run is judged again. Two approved rules are never merged.
    """
    threshold = settings.RULES_MERGE_SIMILARITY if threshold is None else threshold
    Rule.objects.filter(status=Rule.Status.MERGED).update(status=Rule.Status.PENDING, merged_into=None, merge_reason="")
    rules = sorted(Rule.objects.filter(status__in=ACTIVE).select_related("document"), key=_canonical_order)
    if len(rules) < 2:
        return []
    vectors = embed_texts([rule_text(rule) for rule in rules])
    similarity = vectors @ vectors.T
    canonical: list[int] = []
    log: list[dict] = []
    for index, rule in enumerate(rules):
        target = None
        for kept in canonical:
            if rule.status == Rule.Status.APPROVED:
                break
            reason = duplicate_reason(rule, rules[kept], float(similarity[index, kept]), threshold)
            if reason:
                target = (kept, reason)
                break
        if target is None:
            canonical.append(index)
            continue
        kept, reason = target
        into = rules[kept]
        rule.status = Rule.Status.MERGED
        rule.merged_into = into
        rule.merge_reason = reason
        rule.save(update_fields=["status", "merged_into", "merge_reason", "updated_at"])
        log.append({
            "rule": rule.key, "into": into.key, "reason": reason,
            "cosine": round(float(np.clip(similarity[index, kept], -1, 1)), 3),
            "documents": [rule.document.title, into.document.title],
        })
    logger.info("rules merge: %d merged into %d canonical rules", len(log), len(canonical))
    return log
