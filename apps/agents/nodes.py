"""Node functions of the compliance graph. Each takes the state and returns only the keys it changes."""
from __future__ import annotations

import re

from django.conf import settings
from langgraph.types import interrupt

from apps.agents import tools
from apps.agents.schemas import Classification, Draft, Plan, RuleCheck, Verification
from apps.agents.trace import ask, note
from apps.rules.services.text import quote_in_chunk
from core.graph_store import GraphStoreError

MAX_SOURCES = 12
GRAPH_CHUNKS = 3
REFUSAL = (
    "Sorry, I can only answer questions about the maintenance documents in this corpus: airworthiness "
    "directives, maintenance procedures, parts, intervals and the extracted rules. Please ask about one of those."
)
ACKNOWLEDGES_CONFLICT = re.compile(r"conflict|contradict|disagree|inconsisten|do not agree|differ", re.IGNORECASE)


def _tools(state) -> list[str]:
    return state.get("plan", {}).get("tools", [])


def _add_sources(existing: list[dict], new: list[dict], loop: int) -> tuple[list[dict], int]:
    merged = list(existing)
    seen = {row["chunk_id"] for row in merged}
    added = 0
    for row in new:
        if row["chunk_id"] in seen or len(merged) >= MAX_SOURCES:
            continue
        merged.append({**row, "loop": loop})
        seen.add(row["chunk_id"])
        added += 1
    return merged, added


def classify(state):
    out = ask(Classification, "agent_classify", {"question": state["question"]}, "agent.classify")
    note(reason=out.reason)
    return {"intent": out.intent, "entities": out.entities, "loops": 0, "query": state["question"]}


def route_after_classify(state) -> str:
    return "refuse" if state["intent"] == "out_of_scope" else "plan"


def refuse(state):
    return {"final": {"status": "refused", "answer": REFUSAL, "citations": [], "conflicts": [], "warning": ""}}


def plan(state):
    out = ask(
        Plan, "agent_plan",
        {"question": state["question"], "intent": state["intent"], "entities": ", ".join(state.get("entities", [])) or "(none)"},
        "agent.plan",
    )
    chosen = list(out.tools)
    added = []
    if state["intent"] in ("compliance_check", "conflict_check") and "rule_lookup" not in chosen:
        chosen.append("rule_lookup")
        added.append("rule_lookup (compliance and conflict questions always check the rules)")
    note(reason=out.reason, added_by_policy=added)
    return {"plan": {"tools": chosen, "search_query": out.search_query, "reason": out.reason},
            "query": out.search_query}


def hybrid_retrieve(state):
    loop = state.get("loops", 0)
    result = tools.hybrid_search(query=state["query"], k=settings.AGENT_TOP_K)
    hits, added = _add_sources(state.get("hits", []), result["hits"], loop)
    note(new_sources=added, weak_context=result["weak"], mode=result["mode"])
    return {"hits": hits}


def graph_lookup(state):
    if "kg_query" not in _tools(state):
        note(skipped="kg_query not in plan")
        return {"facts": []}
    question = state["question"] if not state.get("loops") else f"{state['question']} {state['query']}"
    try:
        result = tools.kg_query(question=question)
    except GraphStoreError as exc:
        note(error=f"graph store unavailable: {exc}")
        return {"facts": [], "graph_error": str(exc)}
    chunk_ids: list[int] = []
    for fact in result["facts"]:
        for chunk_id in fact["chunk_ids"]:
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
    hits, added = _add_sources(state.get("hits", []), tools.sources_for_chunks(chunk_ids[:GRAPH_CHUNKS], "graph"),
                               state.get("loops", 0))
    note(seeds=[seed["name"] for seed in result["seeds"]], new_sources=added)
    return {"facts": result["facts"], "hits": hits}


def _rules_block(rules: list[dict]) -> str:
    lines = []
    for row in rules:
        exception = f" EXCEPTION {row['exception']}" if row["exception"] else ""
        flags = f" [CONFLICTS WITH {', '.join(c['other'] for c in row['conflicts'])}]" if row["conflicts"] else ""
        lines.append(f"[{row['key']}] ({row['status']}) IF {row['condition']} THEN {row['action']}{exception}"
                     f" - {row['citation']}{flags}")
    return "\n".join(lines)


def rule_check(state):
    if "rule_lookup" not in _tools(state):
        note(skipped="rule_lookup not in plan")
        return {"rules": [], "rule_check": {"skipped": True, "facts": [], "assessments": [], "applicable": [], "conflicts": []}}
    previous = state.get("rule_check") or {}
    if state.get("loops") and previous.get("done"):
        note(reused="rule check from the first pass; the question and its facts did not change")
        return {}
    candidates = tools.rule_lookup(question=state["question"], entities=state.get("entities", []))
    if not candidates:
        note(candidates=0)
        return {"rules": [], "rule_check": {"done": True, "facts": [], "assessments": [], "applicable": [], "conflicts": []}}
    out = ask(RuleCheck, "agent_rule_check",
              {"question": state["question"], "rules": _rules_block(candidates)}, "agent.rule_check")
    known = {row["key"]: row for row in candidates}
    assessed = {item.rule_key: item for item in out.assessments if item.rule_key in known}
    ignored = sorted({item.rule_key for item in out.assessments} - set(known))
    assessments = []
    for key, row in known.items():
        item = assessed.get(key)
        assessments.append({"rule": key, "applies": item.applies if item else "unclear",
                            "relevant": item.relevant if item else False,
                            "reason": item.reason if item else "not assessed by the model",
                            "citation": row["citation"]})
    applicable = [row["rule"] for row in assessments if row["applies"] == "yes" or row["relevant"]]
    possible = [row["rule"] for row in assessments if row["applies"] == "unclear" and row["rule"] not in applicable]
    conflicts = []
    seen_pairs = set()
    for key in applicable:
        for conflict in known[key]["conflicts"]:
            pair = tuple(sorted((key, conflict["other"])))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            other = known.get(conflict["other"], {})
            conflicts.append({
                "rule": key, "rule_text": f"IF {known[key]['condition']} THEN {known[key]['action']}",
                "rule_citation": known[key]["citation"], "rule_chunk_id": known[key]["chunk_id"],
                "other": conflict["other"],
                "other_text": f"IF {other.get('condition', '')} THEN {other.get('action', '')}",
                "other_citation": conflict["other_citation"], "other_chunk_id": other.get("chunk_id"),
                "severity": conflict["severity"], "reason": conflict["reason"],
            })
    rule_chunks = [known[key]["chunk_id"] for key in applicable + possible]
    rule_chunks += [c["other_chunk_id"] for c in conflicts if c["other_chunk_id"]]
    hits, added = _add_sources(state.get("hits", []), tools.sources_for_chunks(list(dict.fromkeys(rule_chunks)), "rule"),
                               state.get("loops", 0))
    note(candidates=len(candidates), ignored_unknown_keys=ignored, new_sources=added)
    return {"rules": candidates, "hits": hits,
            "rule_check": {"done": True, "facts": out.facts, "assessments": assessments,
                           "applicable": applicable, "possible": possible, "conflicts": conflicts}}


def _sources_block(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[S{index}] {row['citation']} (via {row['origin']})\n{row['text']}" for index, row in enumerate(hits, start=1)
    ) or "(no sources)"


def _source_number(hits: list[dict], chunk_id) -> str:
    for index, row in enumerate(hits, start=1):
        if row["chunk_id"] == chunk_id:
            return f"S{index}"
    return "not retrieved"


def _conflicts_block(conflicts: list[dict], hits: list[dict]) -> str:
    if not conflicts:
        return "(none)"
    return "\n".join(
        f"- {c['rule']} ({_source_number(hits, c['rule_chunk_id'])}, {c['rule_citation']}): {c['rule_text']}\n"
        f"  vs {c['other']} ({_source_number(hits, c['other_chunk_id'])}, {c['other_citation']}): {c['other_text']}\n"
        f"  judge: {c['reason']}"
        for c in conflicts
    )


def draft(state):
    hits = state.get("hits", [])
    check = state.get("rule_check") or {}
    applicable = set(check.get("applicable", [])) | set(check.get("possible", []))
    facts_block = "\n".join(
        f"[F{index}] {fact['source']} {fact['relation']} {fact['target']} - {fact['citation']}"
        for index, fact in enumerate(state.get("facts", []), start=1)
    ) or "(none)"
    rules_block = "\n".join(
        f"[{row['rule']}] applies={row['applies']} relevant={row['relevant']}: {row['reason']} - {row['citation']}"
        for row in check.get("assessments", []) if row["rule"] in applicable
    ) or "(none)"
    feedback = "(first draft)"
    if state.get("loops") and state.get("verification"):
        feedback = "\n".join(f"- {problem}" for problem in state["verification"]["problems"])
    out = ask(Draft, "agent_draft", {
        "question": state["question"],
        "sources": _sources_block(hits),
        "facts": facts_block,
        "rules": rules_block,
        "conflicts": _conflicts_block(check.get("conflicts", []), hits),
        "feedback": feedback,
    }, "agent.draft")
    note(sources=len(hits), claims=len(out.claims))
    return {"draft": out.model_dump()}


def verify_citations(state):
    hits = state.get("hits", [])
    current = state["draft"]
    conflicts = (state.get("rule_check") or {}).get("conflicts", [])
    checks = []
    problems = []
    for index, claim in enumerate(current["claims"], start=1):
        row = {"claim": index, "source": f"S{claim['source']}", "in_retrieved_set": 1 <= claim["source"] <= len(hits)}
        row["quote_found"] = row["in_retrieved_set"] and quote_in_chunk(claim["quote"], hits[claim["source"] - 1]["text"])
        if not row["in_retrieved_set"]:
            problems.append(f"claim {index} cites S{claim['source']}, which is not in the retrieved set")
        elif not row["quote_found"]:
            problems.append(f"claim {index}: the quote is not in S{claim['source']} ({hits[claim['source'] - 1]['citation']})")
        checks.append(row)
    to_judge = [row["claim"] for row in checks if row["quote_found"]]
    refined = ""
    if to_judge:
        claims_block = "\n\n".join(
            f"[C{number}] CLAIM: {current['claims'][number - 1]['text']}\n"
            f"QUOTE: {current['claims'][number - 1]['quote']}\n"
            f"SOURCE TEXT ({hits[current['claims'][number - 1]['source'] - 1]['citation']}):\n"
            f"{hits[current['claims'][number - 1]['source'] - 1]['text']}"
            for number in to_judge
        )
        judged = ask(Verification, "agent_verify", {"question": state["question"], "claims": claims_block}, "agent.verify")
        refined = judged.refined_query.strip()
        verdicts = {item.claim: item for item in judged.verdicts}
        for row in checks:
            if not row["quote_found"]:
                continue
            verdict = verdicts.get(row["claim"])
            row["supported"] = bool(verdict and verdict.supported)
            row["judge"] = verdict.reason if verdict else "the judge returned no verdict for this claim"
            if not row["supported"]:
                problems.append(f"claim {row['claim']} is not supported by its quote: {row['judge']}")
    if conflicts and not ACKNOWLEDGES_CONFLICT.search(current["answer"]):
        problems.append("an applicable rule has a conflict ("
                        + "; ".join(f"{c['rule']} vs {c['other']}" for c in conflicts)
                        + ") and the answer does not say so")
    passed = not problems
    loops = state.get("loops", 0)
    verification = {"passed": passed, "checks": checks, "problems": problems, "refined_query": refined,
                    "judged": len(to_judge)}
    if not passed and loops < settings.AGENT_MAX_LOOPS:
        query = refined or f"{state['question']} {' '.join(state.get('entities', []))}".strip()
        note(decision=f"re-retrieve (loop {loops + 1} of {settings.AGENT_MAX_LOOPS})", next_query=query)
        return {"verification": verification, "retry": True, "loops": loops + 1, "query": query}
    warning = ""
    if not passed:
        warning = (f"Citation check still failing after {loops} re-retrieval(s): " + "; ".join(problems)
                   + ". Review the sources before approving.")
    note(decision="to approval" + (" with a warning" if warning else ""))
    return {"verification": verification, "retry": False, "warning": warning}


def route_after_verify(state) -> str:
    return "hybrid_retrieve" if state.get("retry") else "human_approval"


def human_approval(state):
    decision = interrupt({
        "answer": state["draft"]["answer"],
        "verified": state["verification"]["passed"],
        "warning": state.get("warning", ""),
    })
    return {"approval": {"approved": bool(decision.get("approved")), "note": str(decision.get("note", ""))[:1000]}}


def final(state):
    hits = state.get("hits", [])
    current = state["draft"]
    checks = {row["claim"]: row for row in state["verification"]["checks"]}
    citations = []
    for index, claim in enumerate(current["claims"], start=1):
        source = hits[claim["source"] - 1] if 1 <= claim["source"] <= len(hits) else None
        check = checks.get(index, {})
        citations.append({
            "n": index,
            "claim": claim["text"],
            "quote": claim["quote"],
            "document": source["document"] if source else "",
            "page": source["page"] if source else None,
            "section": source["section"] if source else "",
            "citation": source["citation"] if source else f"S{claim['source']} (not retrieved)",
            "verified": bool(check.get("quote_found") and check.get("supported")),
        })
    approval = state.get("approval", {})
    return {"final": {
        "status": "approved" if approval.get("approved") else "rejected",
        "draft_status": current["status"],
        "answer": current["answer"],
        "conflict_note": current.get("conflict_note", ""),
        "citations": citations,
        "conflicts": (state.get("rule_check") or {}).get("conflicts", []),
        "verified": state["verification"]["passed"],
        "warning": state.get("warning", ""),
        "approval_note": approval.get("note", ""),
    }}
