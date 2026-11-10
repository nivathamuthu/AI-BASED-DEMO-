"""Offline stand-ins for the agent prompts (LLM_PROVIDER=mock only).

They read the rendered prompt with keyword patterns so tests and the offline demo can drive the
graph. They never parse text produced by a model, and they are never used when a real provider is set.
"""
import re

from apps.agents.schemas import Classification, Draft, Plan, RuleCheck, Verification
from apps.rules.services.text import content_words, identifiers, normalize_quote
from core.mock_llm import register

_DOMAIN = re.compile(
    r"\bAD \d|\bpart\b|\bPN-|\bP/N|side-stay|bushing|flight hours?|C-check|\bMEL\b|\bdefer|inspection|directive"
    r"|procedure|replace|maintenance|\brules?\b|EO-\d+|MPP-",
    re.IGNORECASE,
)
_CONFLICT = re.compile(r"\bconflict|\bcontradict|\bdisagree|\bagree\b|\bconsistent\b|which (?:one|document) wins",
                       re.IGNORECASE)
_COMPLIANCE = re.compile(r"\b(can|may|allowed|permitted|must|due|need to|keep flying|comply|compliant|legal)\b",
                         re.IGNORECASE)
_HOURS = re.compile(r"\b\d{1,3}(?:,\d{3})+|\b\d{3,6}\b(?= ?(?:flight hours|FH))", re.IGNORECASE)


def _after(prompt: str, marker: str, end: str | None = None) -> str:
    start = prompt.rfind(marker)
    if start < 0:
        return ""
    start += len(marker)
    stop = prompt.find(end, start) if end else -1
    return prompt[start:stop if stop >= 0 else None].strip()


def _question(prompt: str) -> str:
    return _after(prompt, "QUESTION:")


@register(Classification)
def mock_classify(prompt: str) -> dict:
    question = _question(prompt)
    entities = sorted(identifiers(question)) + _HOURS.findall(question)
    if not _DOMAIN.search(question):
        return {"intent": "out_of_scope", "entities": [], "reason": "No maintenance-document terms in the question."}
    if _CONFLICT.search(question):
        intent = "conflict_check"
    elif _COMPLIANCE.search(question):
        intent = "compliance_check"
    else:
        intent = "lookup"
    return {"intent": intent, "entities": entities, "reason": f"Keyword routing chose {intent}."}


@register(Plan)
def mock_plan(prompt: str) -> dict:
    question = _question(prompt)
    intent = _after(prompt, "INTENT:", "\n")
    tools = ["hybrid_search"]
    if identifiers(question):
        tools.append("kg_query")
    if intent in ("compliance_check", "conflict_check"):
        tools.append("rule_lookup")
    return {"tools": tools, "search_query": question[:300], "reason": "Search first, then the graph and rules."}


_RULE_LINE = re.compile(r"^\[(R-\d{4})\] \((\w+)\) (.+)$", re.MULTILINE)


@register(RuleCheck)
def mock_rule_check(prompt: str) -> dict:
    question = _question(prompt)
    wanted_ids, wanted_words = identifiers(question), content_words(question)
    assessments = []
    for key, _status, body in _RULE_LINE.findall(_after(prompt, "RULES:", "QUESTION:")):
        if wanted_ids & identifiers(body):
            applies, reason = "yes", "The rule names an identifier from the question."
        elif len(wanted_words & content_words(body)) >= 2:
            applies, reason = "unclear", "The rule shares wording with the question but no identifier."
        else:
            applies, reason = "no", "The rule is about something else."
        assessments.append({"rule_key": key, "applies": applies, "relevant": applies == "yes", "reason": reason})
    facts = [part.strip() for part in re.split(r"(?<=[.?!])\s+", question) if part.strip()][:10]
    return {"facts": facts, "assessments": assessments}


_SOURCE = re.compile(r"^\[S(\d+)\] [^\n]*\n(.*?)(?=^\[S\d+\] |\Z)", re.MULTILINE | re.DOTALL)
_CONFLICT_LINE = re.compile(r"^- (R-\d{4}) \((S\d+|not retrieved)[^\n]*\n\s+vs (R-\d{4}) \((S\d+|not retrieved)",
                            re.MULTILINE)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if len(part.strip()) >= 8]


def _best_sentence(text: str, wanted: set[str], minimum: int = 2) -> tuple[str, int]:
    best, best_score = "", 0
    for sentence in _sentences(text):
        overlap = len(wanted & content_words(sentence))
        if overlap > best_score:
            best, best_score = sentence, overlap
    return (best, best_score) if best_score >= minimum else ("", 0)


@register(Draft)
def mock_draft(prompt: str) -> dict:
    question = _question(prompt)
    wanted = content_words(question) | {item.lower() for item in identifiers(question)}
    sources = {int(number): text for number, text in _SOURCE.findall(_after(prompt, "SOURCES:", "FACTS:"))}
    ranked = []
    for number, text in sources.items():
        for sentence in _sentences(text):
            overlap = len(wanted & content_words(sentence))
            if overlap >= 2:
                ranked.append((-overlap, number, sentence))
    claims = []
    for _score, number, sentence in sorted(ranked)[:2]:
        if all(claim["quote"] != sentence for claim in claims):
            claims.append({"text": sentence, "source": number, "quote": sentence})
    conflicts = _CONFLICT_LINE.findall(_after(prompt, "CONFLICTS:", "FEEDBACK:"))
    conflict_note = ""
    sides = []
    for rule, rule_source, other, other_source in conflicts:
        for key, label in ((rule, rule_source), (other, other_source)):
            if label.startswith("S") and int(label[1:]) in sources:
                sentence, _score = _best_sentence(sources[int(label[1:])], wanted, minimum=1)
                if sentence and all(claim["quote"] != sentence for claim in claims):
                    claims.append({"text": f"{key}: {sentence}", "source": int(label[1:]), "quote": sentence})
                    sides.append(f"{key} says \"{sentence}\"")
        conflict_note = f"{rule} and {other} conflict."
    if not claims:
        return {"status": "cannot_determine", "answer": "The retrieved sources do not answer this question.",
                "claims": [], "conflict_note": ""}
    answer = " ".join(claim["text"] for claim in claims[:2])
    if conflict_note:
        answer = "The documents conflict, so I cannot pick one: " + "; ".join(sides) + "."
    return {"status": "answered", "answer": answer, "claims": claims[:8], "conflict_note": conflict_note}


_CLAIM = re.compile(r"^\[C(\d+)\] CLAIM: (.*?)\nQUOTE: (.*?)\nSOURCE TEXT", re.MULTILINE | re.DOTALL)


@register(Verification)
def mock_verify(prompt: str) -> dict:
    verdicts = []
    for number, claim, quote in _CLAIM.findall(_after(prompt, "CLAIMS:", "QUESTION:")):
        supported = normalize_quote(quote) in normalize_quote(claim) or bool(
            content_words(claim) & content_words(quote))
        verdicts.append({"claim": int(number), "supported": supported,
                         "reason": "The quote states the claim." if supported else "The quote does not state the claim."})
    return {"verdicts": verdicts, "refined_query": ""}
