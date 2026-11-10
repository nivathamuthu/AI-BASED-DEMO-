"""Offline stand-ins for the KG prompts (LLM_PROVIDER=mock).

The extractor reads the source chunk with a small aviation lexicon and links
entities that share a sentence. It plays the part of the model for tests and
the offline demo; it never parses text produced by a model.

v2 ("only extract what the text states") keeps every relation inside the
chunk. v1 behaves like a naive prompt: it names endpoints loosely ("the
aircraft") and adds one relation from background knowledge, so the endpoint
check has something to drop.
"""
import re

from apps.kg.schemas import GraphAnswer, KGOutput
from core.mock_llm import register

_PATTERNS: list[tuple[str, str]] = [
    ("Regulation", r"\bAD \d{4}-\d{2}-\d{2}\b"),
    ("Regulation", r"\bMEL\b"),
    ("Part", r"\bPN-\d{2}-\d{4}\b|\bP/N ?\d{2}-\d{4}\b|\b\d{2}-\d{5}-\d\b"),
    ("Procedure", r"\bMPP-[A-Z]+-\d{4}\b|\bcomponent maintenance manual\b"),
    ("Interval", r"\b\d{2,6} flight hours?\b|\b\d{4,6}\b|\bC-check\b"),
    ("Aircraft", r"\bnarrow-body fleet\b|\bwide-body fleet\b"),
    ("Component", r"\bmain landing gear side-stay\b|\bside-stay\b|\bbushing\b|\bpin\b"),
    ("Task", r"\bdetailed visual inspection\b|\binspection\b|\breplace(?:ment|d|s)?\b|\bdeferral\b"),
]
_COMPILED = [(kind, re.compile(pattern, re.IGNORECASE)) for kind, pattern in _PATTERNS]
_TASK_NAMES = {"replace": "Replacement", "inspection": "Detailed visual inspection", "deferral": "Deferral"}


def _between(prompt: str, start: str, end: str | None) -> str:
    begin = prompt.rfind(start)
    if begin < 0:
        return ""
    begin += len(start)
    finish = prompt.find(end, begin) if end else -1
    return prompt[begin:finish if finish >= 0 else None].strip()


def _source_chunk(prompt: str) -> str:
    if "CHUNK:" in prompt:
        return _between(prompt, "CHUNK:", None)
    lines = prompt.splitlines()
    return "\n".join(lines[1:]).strip()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def _canonical(kind: str, surface: str) -> tuple[str, list[str]]:
    """Display name plus the surface form as an alias when they differ."""
    text = " ".join(surface.split())
    if kind == "Task":
        lowered = text.lower()
        for stem, name in _TASK_NAMES.items():
            if lowered.startswith(stem) or lowered == stem or (stem == "inspection" and "inspection" in lowered):
                return name, [] if lowered == name.lower() else [text]
    if kind == "Interval" and text.isdigit():
        return f"{text} flight hours", [text]
    if kind == "Interval" and text.lower().endswith("flight hour"):
        return f"{text}s", [text]
    if kind == "Component" and text.lower() == "main landing gear side-stay":
        return "side-stay", [text]
    return text, []


def _mentions(sentence: str) -> list[tuple[int, str, str, list[str]]]:
    """Non-overlapping (position, type, name, aliases), longest match first."""
    found = []
    about_hours = "hour" in sentence.lower()
    for kind, pattern in _COMPILED:
        for match in pattern.finditer(sentence):
            if kind == "Interval" and match.group(0).isdigit() and not about_hours:
                continue
            if kind == "Task" and match.group(0).isupper():
                continue
            found.append((match.start(), match.end(), kind, match.group(0)))
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    taken: list[tuple[int, int]] = []
    mentions = []
    for start, end, kind, surface in found:
        if any(start < used_end and end > used_start for used_start, used_end in taken):
            continue
        taken.append((start, end))
        name, aliases = _canonical(kind, surface)
        mentions.append((start, kind, name, aliases))
    return mentions


def _relations(sentence: str, mentions, carried_regulation: str | None) -> list[dict]:
    by_type: dict[str, list[str]] = {}
    for _position, kind, name, _aliases in mentions:
        names = by_type.setdefault(kind, [])
        if name not in names:
            names.append(name)
    regulations = by_type.get("Regulation", [])
    if not regulations and carried_regulation and re.match(r"^(It|This directive)\b", sentence):
        regulations = [carried_regulation]
    out = []

    def add(source, relation, target):
        if source != target:
            out.append({"source": source, "relation": relation, "target": target, "evidence_quote": sentence})

    for regulation in regulations:
        for kind in ("Aircraft", "Part", "Component"):
            for target in by_type.get(kind, []):
                add(regulation, "APPLIES_TO", target)
    for task in by_type.get("Task", []):
        for regulation in regulations:
            add(task, "GOVERNED_BY", regulation)
        for interval in by_type.get("Interval", []):
            add(task, "HAS_INTERVAL", interval)
        for kind in ("Part", "Component"):
            for target in by_type.get(kind, []):
                add(task, "APPLIES_TO", target)
    for procedure in by_type.get("Procedure", []):
        for task in by_type.get("Task", []):
            add(procedure, "REQUIRES", task)
        for regulation in regulations:
            add(procedure, "GOVERNED_BY", regulation)
    parts = by_type.get("Part", [])
    replace_with = re.search(r"\breplac\w*\b.*?\bwith\b", sentence, re.IGNORECASE)
    if replace_with:
        old = [name for position, kind, name, _a in mentions if kind == "Part" and position < replace_with.end()]
        new = [name for position, kind, name, _a in mentions if kind == "Part" and position >= replace_with.end()]
        for newer in new:
            for older in old:
                add(newer, "REPLACES", older)
    components = by_type.get("Component", [])
    if components:
        for part in parts:
            add(part, "PART_OF", components[0])
    return out


def extract(chunk: str, engineered: bool) -> dict:
    entities: dict[str, dict] = {}
    relations: list[dict] = []
    carried: str | None = None
    for sentence in _sentences(chunk):
        mentions = _mentions(sentence)
        for _position, kind, name, aliases in mentions:
            entry = entities.setdefault(name, {"name": name, "type": kind, "aliases": []})
            for alias in aliases:
                if alias not in entry["aliases"] and alias != name:
                    entry["aliases"].append(alias)
        relations.extend(_relations(sentence, mentions, carried))
        regulations = [name for _p, kind, name, _a in mentions if kind == "Regulation" and name != "MEL"]
        if regulations:
            carried = regulations[0]
    if not engineered:
        for relation in relations:
            if entities.get(relation["target"], {}).get("type") == "Aircraft":
                relation["target"] = "the aircraft"
        if entities:
            first = next(iter(entities))
            relations.append({
                "source": first,
                "relation": "APPLIES_TO",
                "target": "Boeing 737",
                "evidence_quote": "Typical narrow-body type.",
            })
        for entry in entities.values():
            entry["aliases"] = []
    unique = []
    seen = set()
    for relation in relations:
        key = (relation["source"], relation["relation"], relation["target"])
        if key not in seen:
            seen.add(key)
            unique.append(relation)
    return {"entities": list(entities.values())[:60], "relations": unique[:80]}


@register(KGOutput)
def mock_extract(prompt: str) -> dict:
    engineered = "only extract what the text states" in prompt.lower()
    return extract(_source_chunk(prompt), engineered)


_FACT = re.compile(r"^\[F(\d+)\] (.+?) \((\w+)\) (\w+) (.+?) \((\w+)\)", re.MULTILINE)
_WANTED = [
    ("Task", r"\btasks?\b|\bwhat must be done\b|\bwhich actions?\b"),
    ("Interval", r"\bintervals?\b|\bhow many flight hours\b|\blimits?\b|\bwhen\b"),
    ("Part", r"\bwhich parts?\b|\bwhat replaces\b|\breplacement part\b|\bpart numbers?\b"),
    ("Component", r"\bcomponents?\b|\bassembly\b"),
    ("Aircraft", r"\bfleet\b|\baircraft\b"),
    ("Procedure", r"\bprocedures?\b|\bprocedure codes?\b"),
    ("Regulation", r"\bdirectives?\b|\bregulations?\b"),
]
_STOP = {"what", "which", "when", "does", "the", "for", "and", "are", "with", "that", "from", "this", "part"}


def _wanted_type(question: str) -> str | None:
    best = None
    for kind, pattern in _WANTED:
        match = re.search(pattern, question, re.IGNORECASE)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), kind)
    return best[1] if best else None


@register(GraphAnswer)
def mock_graph_answer(prompt: str) -> dict:
    """Name every fact endpoint of the asked-for type that is not already in the question."""
    question = _between(prompt, "QUESTION:", None)
    facts_block = _between(prompt, "FACTS:", "CHUNKS:")
    wanted = _wanted_type(question)
    lowered = question.lower()
    names: list[str] = []
    used: list[int] = []
    for match in _FACT.finditer(facts_block):
        index = int(match.group(1))
        for name, kind in ((match.group(2), match.group(3)), (match.group(5), match.group(6))):
            if kind != wanted or name.lower() in lowered:
                continue
            if name not in names:
                names.append(name)
            if index not in used:
                used.append(index)
    if names:
        return {"answer": f"{wanted}: " + "; ".join(names) + ".", "known": True, "fact_indexes": used, "citation_indexes": [1]}
    chunks = _between(prompt, "CHUNKS:", "QUESTION:")
    wanted_tokens = {t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", lowered) if t not in _STOP and len(t) > 2}
    best, best_overlap = "", 0
    for sentence in _sentences(re.sub(r"\(source:[^)]*\)|\[\d+\]", " ", chunks)):
        tokens = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", sentence.lower()))
        overlap = len(wanted_tokens & tokens)
        if overlap > best_overlap:
            best, best_overlap = sentence, overlap
    if not best:
        return {"answer": "unknown", "known": False, "fact_indexes": [], "citation_indexes": []}
    return {"answer": best[:800], "known": True, "fact_indexes": [], "citation_indexes": [1]}
