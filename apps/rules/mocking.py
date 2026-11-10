"""Offline stand-ins for the rule prompts (LLM_PROVIDER=mock).

The extractor reads the source chunk sentence by sentence with a few clause
patterns. It plays the part of the model for tests and the offline demo; it
never parses text produced by a model. Its quality is whatever these patterns
give, so mock eval scores describe this stand-in, not a prompt.

v2 ("only extract rules the text actually states") copies each rule sentence
verbatim as the quote, attaches a following "Exception:" sentence, and skips
descriptions. v1 behaves like a naive prompt: it turns most sentences into a
rule, uses "always" when it finds no condition, adds an invented deadline to
permissions, and paraphrases quotes that mention flight hours (so the quote
check drops them).
"""
import re

from apps.rules.schemas import ConflictJudgment, RuleSet
from apps.rules.services.text import content_words, identifiers, jaccard, thresholds
from core.mock_llm import register

_MODAL = re.compile(
    r"\b(may be|may stay|must|shall|should|cannot|can not|is not permitted|does not apply|applies|closes)\b",
    re.IGNORECASE,
)
_MARKER = re.compile(r"\s(when|if|while|before)\s", re.IGNORECASE)
_IMPERATIVE = {"replace", "do", "start", "record", "inspect", "torque", "quote", "use", "give", "ground"}
_HEADING_PREFIX = re.compile(r"^.*?\b\d+(?:\.\d+)+ [A-Z][a-z]+ (?=[A-Z])")
_NEGATION = re.compile(r"\b(cannot|can not|not|never|no longer)\b", re.IGNORECASE)


def _source_chunk(prompt: str) -> str:
    marker = prompt.rfind("CHUNK:")
    if marker >= 0:
        return prompt[marker + len("CHUNK:"):].strip()
    return "\n".join(prompt.splitlines()[1:]).strip()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def _clean(text: str) -> str:
    return text.strip().strip(",;").rstrip(".").strip()


def _split_if_then(sentence: str) -> tuple[str, str] | None:
    match = re.match(r"^IF\s+(.+?),?\s+THEN\s+(.+)$", sentence)
    if not match:
        return None
    return _clean(match.group(1)), _clean(match.group(2))


def _split_marker(sentence: str) -> tuple[str, str] | None:
    """'<action> when <condition>' -> (condition, action)."""
    match = _MARKER.search(sentence)
    if not match:
        return None
    action = _clean(sentence[:match.start()])
    condition = _clean(re.split(r",\s+so\b", sentence[match.end():])[0])
    if not action or not condition:
        return None
    return condition, action


def _v2_rule(sentence: str) -> dict | None:
    sentence = _HEADING_PREFIX.sub("", sentence)
    first = sentence.split()[0].lower() if sentence.split() else ""
    pair = _split_if_then(sentence)
    confidence = 0.9
    if pair is None and (first in _IMPERATIVE or _MODAL.search(sentence)):
        pair = _split_marker(sentence)
        confidence = 0.8
    if pair is None and first in _IMPERATIVE:
        match = re.search(r"\s(at (?:every|the next|\d).*)$", sentence)
        if match:
            pair = (_clean(match.group(1)), _clean(sentence[:match.start()]))
            confidence = 0.75
    if pair is None:
        modal = _MODAL.search(sentence)
        if modal and modal.start() > 0 and first not in _IMPERATIVE:
            pair = (_clean(sentence[:modal.start()]), _clean(sentence[modal.start():]))
            confidence = 0.7
    if pair is None:
        return None
    condition, action = pair
    return {"condition": condition, "action": action, "exception": "", "source_quote": sentence, "confidence": confidence}


def _v1_rule(sentence: str) -> dict | None:
    if len(sentence.split()) < 5:
        return None
    pair = _split_if_then(sentence) or _split_marker(sentence)
    condition, action = pair if pair else ("always", _clean(sentence))
    if _MODAL.search(sentence) and not thresholds(sentence):
        action = f"{action} within 30 days"
    quote = sentence.replace("flight hours", "FH")
    return {"condition": condition, "action": action, "exception": "", "source_quote": quote, "confidence": 0.95}


def extract(chunk: str, engineered: bool) -> dict:
    rules: list[dict] = []
    for sentence in _sentences(chunk):
        if engineered and sentence.startswith("Exception:"):
            if rules:
                rules[-1]["exception"] = _clean(sentence[len("Exception:"):])
            continue
        rule = _v2_rule(sentence) if engineered else _v1_rule(sentence)
        if rule:
            rules.append(rule)
    return {"rules": rules[:25]}


@register(RuleSet)
def mock_extract(prompt: str) -> dict:
    engineered = "only extract rules the text actually states" in prompt.lower()
    return extract(_source_chunk(prompt), engineered)


def _field(block: str, name: str) -> str:
    match = re.search(rf"^{name}: (.*)$", block, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _rule_block(prompt: str, label: str, next_label: str | None) -> dict:
    start = prompt.rfind(f"{label} (")
    end = prompt.find(f"{next_label} (", start) if next_label else -1
    block = prompt[start:end if end >= 0 else None] if start >= 0 else ""
    return {name: _field(block, name) for name in ("condition", "action", "exception", "quote")}


# Main verb of an action, most specific first: "the replacement may be deferred" is a deferral.
_VERBS = ("defer", "ferry", "inspect", "record", "replac", "appl")


def _verb(text: str) -> str:
    lowered = text.lower()
    return next((stem for stem in _VERBS if stem in lowered), "")


@register(ConflictJudgment)
def mock_judge(prompt: str) -> dict:
    """Same identifier and action verb with different thresholds, or the same subject negated."""
    a = _rule_block(prompt, "RULE A", "RULE B")
    b = _rule_block(prompt, "RULE B", None)
    text_a = f"{a['condition']} {a['action']}"
    text_b = f"{b['condition']} {b['action']}"
    shared = identifiers(text_a) & identifiers(text_b)
    numbers_a, numbers_b = thresholds(text_a), thresholds(text_b)
    same_verb = bool(_verb(a["action"])) and _verb(a["action"]) == _verb(b["action"])
    if shared and same_verb and numbers_a and numbers_b and not numbers_a & numbers_b:
        item = min(shared)
        return {
            "conflict": True, "kind": "threshold", "severity": "high",
            "reason": f"{item}: one rule says {', '.join(sorted(numbers_a))}, the other {', '.join(sorted(numbers_b))}.",
        }
    negated_a = bool(_NEGATION.search(a["action"]))
    negated_b = bool(_NEGATION.search(b["action"]))
    subject = jaccard(content_words(a["condition"]), content_words(b["condition"]))
    if negated_a != negated_b and same_verb and subject >= 0.5:
        return {
            "conflict": True, "kind": "action", "severity": "medium",
            "reason": f"Same condition, but one action is negated: '{a['action']}' vs '{b['action']}'.",
        }
    return {"conflict": False, "kind": "none", "severity": "low", "reason": "Different items or compatible actions."}
