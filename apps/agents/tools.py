"""The agent's three tools: plain functions over Modules 3, 4 and 5, with typed keyword arguments."""
from __future__ import annotations

from django.conf import settings
from django.db.models import Q

from apps.agents.trace import tool
from apps.ingest.models import Chunk
from apps.kg.services.graphrag import link_entities, rank_facts
from apps.rag.services.hits import Hit
from apps.rag.services.retrieve import is_weak, retrieve
from apps.rules.models import Rule, RuleConflict
from apps.rules.services.text import content_words, identifiers
from core.graph_store import get_store


def source_from_hit(hit: Hit, origin: str) -> dict:
    return {
        "chunk_id": hit.chunk_id,
        "document": hit.document_title,
        "page": hit.page_start,
        "section": hit.section,
        "citation": hit.citation(),
        "text": hit.text,
        "score": hit.score,
        "origin": origin,
    }


def sources_for_chunks(chunk_ids: list[int], origin: str) -> list[dict]:
    rows = {chunk.id: chunk for chunk in Chunk.objects.select_related("document").filter(id__in=chunk_ids)}
    sources = []
    for chunk_id in chunk_ids:
        chunk = rows.get(chunk_id)
        if chunk is not None:
            hit = Hit(chunk_id=chunk.id, text=chunk.text, document_id=chunk.document_id,
                      document_title=chunk.document.title, page_start=chunk.page_start,
                      section=chunk.section_path, score=1.0, lexical=1)
            sources.append(source_from_hit(hit, origin))
    return sources


def _summarize_search(result: dict) -> dict:
    return {"mode": result["mode"], "weak": result["weak"],
            "hits": [f"#{row['chunk_id']} {row['citation']} ({row['score']})" for row in result["hits"]]}


@tool(_summarize_search)
def hybrid_search(query: str, k: int = 6, mode: str = "") -> dict:
    """Module 3 retrieval in the mode that scored best on eval_rag (AGENT_RETRIEVAL_MODE)."""
    mode = mode or settings.AGENT_RETRIEVAL_MODE
    hits = retrieve(query, mode, k=k)
    return {"mode": mode, "weak": is_weak(hits, mode), "hits": [source_from_hit(hit, "search") for hit in hits]}


def _summarize_graph(result: dict) -> dict:
    return {"seeds": [f"{seed['name']} ({seed['reason']})" for seed in result["seeds"]],
            "facts": [f"{fact['source']} {fact['relation']} {fact['target']}" for fact in result["facts"]]}


@tool(_summarize_graph)
def kg_query(question: str, hops: int = 2, limit: int = 8) -> dict:
    """Module 4: link question entities to graph nodes, walk `hops` hops, rank the facts."""
    store = get_store()
    seeds = link_entities(question, store.nodes())
    facts: list[dict] = []
    if seeds:
        nodes, edges = store.neighborhood([seed.id for seed in seeds], hops=hops)
        facts = rank_facts(question, seeds, nodes, edges)[:limit]
    return {"seeds": [{"name": seed.name, "type": seed.type, "reason": seed.reason} for seed in seeds],
            "facts": facts}


def _summarize_rules(result: list[dict]) -> list[str]:
    return [f"{row['key']} ({row['status']}, score {row['score']}"
            f"{', conflicts ' + ', '.join(c['other'] for c in row['conflicts']) if row['conflicts'] else ''})"
            for row in result]


def _rule_row(rule: Rule, score: int, conflicts: list[dict], reason: str) -> dict:
    return {
        "key": rule.key,
        "status": rule.status,
        "condition": rule.condition,
        "action": rule.action,
        "exception": rule.exception,
        "quote": rule.source_quote,
        "chunk_id": rule.chunk_id,
        "citation": rule.citation(),
        "score": score,
        "matched_by": reason,
        "conflicts": conflicts,
    }


def _conflicts_by_rule(rule_ids: set[int]) -> dict[int, list[dict]]:
    rows = RuleConflict.objects.filter(is_conflict=True).filter(
        Q(rule_a_id__in=rule_ids) | Q(rule_b_id__in=rule_ids)
    ).select_related("rule_a__document", "rule_b__document")
    found: dict[int, list[dict]] = {}
    for row in rows:
        for mine, other in ((row.rule_a, row.rule_b), (row.rule_b, row.rule_a)):
            if mine.id in rule_ids:
                found.setdefault(mine.id, []).append({
                    "other": other.key, "other_id": other.id, "other_citation": other.citation(),
                    "kind": row.kind, "severity": row.severity, "reason": row.reason,
                })
    return found


@tool(_summarize_rules)
def rule_lookup(question: str, entities: list[str], limit: int = 8) -> list[dict]:
    """Module 5: approved and pending rules that share an identifier or wording with the question.

    Each row carries its conflict flags. A conflicting partner rule is added to the list even when it
    did not match, so the answer can show both sides.
    """
    text = " ".join([question, *entities])
    wanted_ids = identifiers(text)
    wanted_words = content_words(text)
    rules = list(Rule.objects.filter(status__in=[Rule.Status.APPROVED, Rule.Status.PENDING]).select_related("document"))
    scored = []
    for rule in rules:
        body = f"{rule.condition} {rule.action} {rule.exception}"
        shared_ids = wanted_ids & identifiers(body)
        overlap = len(wanted_words & content_words(body))
        score = 3 * len(shared_ids) + overlap
        if shared_ids or overlap >= 2:
            reason = f"identifier {', '.join(sorted(shared_ids))}" if shared_ids else f"{overlap} shared words"
            scored.append((score, rule, reason))
    scored.sort(key=lambda item: (-item[0], item[1].key))
    chosen = scored[:limit]
    by_id = {rule.id: rule for rule in rules}
    conflicts = _conflicts_by_rule({rule.id for _score, rule, _reason in chosen})
    rows = [_rule_row(rule, score, conflicts.get(rule.id, []), reason) for score, rule, reason in chosen]
    present = {row["key"] for row in rows}
    for row in list(rows):
        for conflict in row["conflicts"]:
            partner = by_id.get(conflict["other_id"])
            if partner is not None and partner.key not in present:
                present.add(partner.key)
                partner_conflicts = _conflicts_by_rule({partner.id}).get(partner.id, [])
                rows.append(_rule_row(partner, 0, partner_conflicts, f"conflicts with {row['key']}"))
    return rows
