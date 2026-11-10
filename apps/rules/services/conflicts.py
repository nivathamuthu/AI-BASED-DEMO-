"""Find candidate rule pairs cheaply, then ask a validated LLM judge about each candidate only."""
import hashlib
import logging
from itertools import combinations

from django.conf import settings

from apps.rules.models import Rule, RuleConflict
from apps.rules.schemas import ConflictJudgment
from apps.rules.services.merge import ACTIVE
from apps.rules.services.text import identifiers
from core.embeddings import embed_texts
from core.llm import LLMValidationError, describe_backend, structured_call
from core.prompts import render_prompt

logger = logging.getLogger(__name__)

JUDGE_TASK = "judge_rule_conflict"
JUDGE_VERSION = "v1"


def pair_signature(left: Rule, right: Rule) -> str:
    """Changes when either rule's text, the judge prompt version or the judging model changes."""
    backend = describe_backend()
    text = "|".join(
        f"{rule.key}:{rule.condition}:{rule.action}:{rule.exception}" for rule in (left, right)
    )
    judge = f"{JUDGE_VERSION}|{backend['provider']}|{backend['model']}"
    return hashlib.sha256(f"{judge}|{text}".encode()).hexdigest()


def candidate_pairs(rules: list[Rule], threshold: float | None = None) -> list[tuple[Rule, Rule, str]]:
    """Pairs that share a part / directive identifier, or whose conditions are similar."""
    threshold = settings.RULES_CONFLICT_SIMILARITY if threshold is None else threshold
    if len(rules) < 2:
        return []
    vectors = embed_texts([rule.condition for rule in rules])
    similarity = vectors @ vectors.T
    ids = [identifiers(f"{rule.condition} {rule.action}") for rule in rules]
    pairs = []
    for i, j in combinations(range(len(rules)), 2):
        shared = ids[i] & ids[j]
        if shared:
            pairs.append((rules[i], rules[j], f"shared {', '.join(sorted(shared))}"))
        elif similarity[i, j] >= threshold:
            pairs.append((rules[i], rules[j], f"condition cosine {similarity[i, j]:.2f}"))
    return pairs


def _source(rule: Rule) -> str:
    return f"{rule.key}, {rule.citation()}"


def judge_pair(left: Rule, right: Rule) -> ConflictJudgment:
    variables = {
        "domain": "aviation maintenance",
        "source_a": _source(left), "condition_a": left.condition, "action_a": left.action,
        "exception_a": left.exception or "(none)", "quote_a": left.source_quote,
        "source_b": _source(right), "condition_b": right.condition, "action_b": right.action,
        "exception_b": right.exception or "(none)", "quote_b": right.source_quote,
    }
    prompt = render_prompt(JUDGE_TASK, JUDGE_VERSION, variables, ConflictJudgment)
    return structured_call(ConflictJudgment, prompt, task="rules.judge", prompt_name=JUDGE_TASK, prompt_version=JUDGE_VERSION)


def detect_conflicts() -> dict:
    """Judge every candidate pair among active rules. Unchanged pairs reuse the stored verdict."""
    rules = list(Rule.objects.filter(status__in=ACTIVE).select_related("document").order_by("key"))
    active_ids = [rule.id for rule in rules]
    RuleConflict.objects.exclude(rule_a_id__in=active_ids, rule_b_id__in=active_ids).delete()
    pairs = candidate_pairs(rules)
    wanted = {(left.id, right.id) for left, right, _why in pairs}
    stale = [row.id for row in RuleConflict.objects.all() if (row.rule_a_id, row.rule_b_id) not in wanted]
    RuleConflict.objects.filter(id__in=stale).delete()
    stored = {(row.rule_a_id, row.rule_b_id): row for row in RuleConflict.objects.all()}
    backend = describe_backend()
    stats = {"candidates": len(pairs), "judged": 0, "cached": 0, "failed": 0}
    for left, right, why in pairs:
        signature = pair_signature(left, right)
        row = stored.get((left.id, right.id))
        if row and row.signature == signature:
            stats["cached"] += 1
            continue
        try:
            verdict = judge_pair(left, right)
        except LLMValidationError as exc:
            logger.warning("rules judge %s vs %s failed validation: %s", left.key, right.key, exc)
            stats["failed"] += 1
            continue
        stats["judged"] += 1
        RuleConflict.objects.update_or_create(
            rule_a=left,
            rule_b=right,
            defaults={
                "is_conflict": verdict.conflict,
                "signature": signature,
                "kind": verdict.kind,
                "severity": verdict.severity,
                "reason": f"{verdict.reason} (candidate: {why})",
                "llm_provider": backend["provider"],
                "llm_model": backend["model"],
            },
        )
    stats["conflicts"] = RuleConflict.objects.filter(is_conflict=True).count()
    logger.info("rules conflicts: %s", stats)
    return stats
