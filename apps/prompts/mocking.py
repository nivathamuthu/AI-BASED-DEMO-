"""Offline stand-ins so v1 and v2 produce different scores with LLM_PROVIDER=mock.

v2 templates contain instructions the naive v1 template does not. The handlers
key off those instructions. They read the source chunk, not an LLM string.
"""
import re

from core.mock_llm import register

from apps.prompts.schemas import AnswerOutput, RuleOutput

_STOP = {
    "what", "which", "when", "where", "how", "does", "this", "that", "with",
    "from", "have", "been", "will", "your", "into", "about", "long", "specified",
    "required", "number", "part", "side", "stay",
}
_RULE = re.compile(
    r"\bIF\b\s+(.+?)\s+\bTHEN\b\s+(.+?)(?:\s+\bException:\s*(.+))?$",
    re.IGNORECASE | re.DOTALL,
)


def _between(prompt: str, start: str, end: str | None) -> str:
    lower = prompt.lower()
    begin = lower.rfind(start.lower())
    if begin < 0:
        return ""
    begin += len(start)
    if end is None:
        return prompt[begin:].strip()
    finish = lower.find(end.lower(), begin)
    return prompt[begin:finish if finish >= 0 else None].strip()


def _chunk_and_question(prompt: str) -> tuple[str, str]:
    """Pull the chunk and the question out of either the v2 labels or the v1 wording."""
    if "CHUNK:" in prompt:
        return _between(prompt, "CHUNK:", "QUESTION:"), _between(prompt, "QUESTION:", None)
    if "Question:" in prompt:
        head, _, tail = prompt.rpartition("Question:")
        lines = [line.strip() for line in head.splitlines() if line.strip()]
        chunk = "\n".join(lines[1:]) if len(lines) > 1 else head.strip()
        return chunk, tail.strip()
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    return ("\n".join(lines[1:]) if len(lines) > 1 else prompt), ""


def _tokens(text: str) -> set[str]:
    found = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()))
    return {token for token in found if token not in _STOP and len(token) >= 4}


def _sentences(text: str) -> list[str]:
    """Split on sentence ends and line breaks, and drop retrieval labels."""
    cleaned = re.sub(r"\(source:[^)]*\)", " ", text)
    cleaned = re.sub(r"\[\d+\]", " ", cleaned)
    parts = [part.strip(" -") for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if part.strip(" -")]
    return parts or ["unknown"]


@register(AnswerOutput)
def mock_answer(prompt: str) -> dict:
    """v2 abstains when the question's tokens are absent. v1 always answers with sentence one."""
    chunk, question = _chunk_and_question(prompt)
    engineered = "not contained in the chunk" in prompt.lower()
    if not engineered:
        return {"answer": _sentences(chunk)[0][:800], "known": True, "citation_indexes": []}
    wanted = _tokens(question)
    best = ""
    best_overlap = 0
    for sentence in _sentences(chunk):
        overlap = len(wanted & _tokens(sentence))
        if overlap > best_overlap:
            best, best_overlap = sentence, overlap
    if best_overlap == 0:
        return {"answer": "unknown", "known": False, "citation_indexes": []}
    return {"answer": best[:800], "known": True, "citation_indexes": [1]}


@register(RuleOutput)
def mock_rule(prompt: str) -> dict:
    """v2 copies an explicit IF/THEN. v1 invents a rule for every chunk."""
    chunk, _question = _chunk_and_question(prompt)
    engineered = "only when the chunk states" in prompt.lower()
    match = _RULE.search(chunk)
    if engineered and match is None:
        return {"found": False, "condition": "", "action": "", "exception": "", "confidence": 0.0}
    if match is None:
        return {
            "found": True,
            "condition": chunk[:180],
            "action": "follow the procedure",
            "exception": "",
            "confidence": 0.4,
        }
    return {
        "found": True,
        "condition": " ".join(match.group(1).split()),
        "action": " ".join(match.group(2).split()).rstrip("."),
        "exception": " ".join((match.group(3) or "").split()).rstrip("."),
        "confidence": 0.9,
    }
