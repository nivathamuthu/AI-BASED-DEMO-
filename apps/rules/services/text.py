"""Normalization shared by quote verification, merging and conflict candidates."""
import hashlib
import re
import unicodedata

_PUNCT = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u00a0": " ",
})
_IDENTIFIER = re.compile(
    r"\bAD \d{4}-\d{2}-\d{2}\b|\bPN-\d{2}-\d{4}\b|\bP/N ?\d{2}-\d{4}\b|\b\d{2}-\d{5}-\d\b|\b[A-Z]{2,4}-[A-Z]*-?\d{3,5}\b"
)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_WORD = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")
_STOP = {
    "the", "a", "an", "of", "on", "in", "to", "for", "and", "or", "it", "its", "is", "are", "be",
    "at", "by", "with", "this", "that", "when", "if", "then", "while", "before", "under", "itself",
}


def normalize_quote(text: str) -> str:
    """Unicode-fold quotes and dashes and collapse whitespace. Case is kept."""
    text = unicodedata.normalize("NFKC", text or "").translate(_PUNCT)
    return " ".join(text.split())


def quote_in_chunk(quote: str, chunk_text: str) -> bool:
    """True when the quote appears verbatim in the chunk, ignoring whitespace and typographic quotes."""
    needle = normalize_quote(quote).strip()
    return bool(needle) and needle in normalize_quote(chunk_text)


def normalize_clause(text: str) -> str:
    """Lowercase, folded, punctuation-free form of a condition or action, for exact-match merging."""
    return " ".join(_WORD.findall(normalize_quote(text).lower()))


def fingerprint(condition: str, action: str) -> str:
    key = f"{normalize_clause(condition)}|{normalize_clause(action)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def identifiers(text: str) -> set[str]:
    """Part numbers, directive numbers and document codes, normalized (P/N 77-4418 == PN-77-4418)."""
    found = set()
    for match in _IDENTIFIER.findall(normalize_quote(text)):
        value = match.upper().replace("P/N ", "PN-").replace("P/N", "PN-")
        found.add(value)
    return found


def thresholds(text: str) -> set[str]:
    """Numbers that are not part of an identifier: intervals, limits, counts."""
    stripped = _IDENTIFIER.sub(" ", normalize_quote(text))
    stripped = re.sub(r"\b(?:section|§)\s*\d+(?:\.\d+)*", " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\bS\d+\b", " ", stripped)
    return set(_NUMBER.findall(stripped))


def content_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(normalize_quote(text).lower()) if word not in _STOP and len(word) > 2}


def jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
