"""Turn entity surface names into merge keys.

"PN-77-4418", "P/N 77-4418" and "part number 77-4418" all become "pn 77 4418".
"12000 FH" and "12000 flight hours" both become "12000 flight hours".
"""
import re

_PART_PREFIX = re.compile(r"\b(?:part\s+number|part\s+no\.?|p\s*/\s*n|pn)[\s.:#-]*(?=\d)", re.IGNORECASE)
_AD_PREFIX = re.compile(r"\bairworthiness\s+directive\s+(?=\d)", re.IGNORECASE)
_FLIGHT_HOURS = re.compile(r"\b(?:flight\s+hours?|fh|fhs)\b", re.IGNORECASE)
_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_PUNCT = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    """The same spelling rules as entity_key, applied anywhere in a longer text such as a question."""
    text = " ".join(text.split())
    text = _AD_PREFIX.sub("AD ", text)
    text = _PART_PREFIX.sub("pn ", text)
    text = _FLIGHT_HOURS.sub("flight hours", text)
    return _PUNCT.sub(" ", text.lower()).strip()


def entity_key(name: str) -> str:
    """Lowercase, strip articles and punctuation, unify part-number and flight-hour spellings."""
    return normalize_text(_ARTICLE.sub("", " ".join(name.split())))


def has_digit(key: str) -> bool:
    return any(char.isdigit() for char in key)


def identifier_tokens(name: str) -> tuple[str, ...]:
    """The digit-bearing tokens of the merge key: "P/N 77-4418" -> ("77", "4418"), "S1000 through S1899" -> ("s1000", "s1899")."""
    return tuple(token for token in entity_key(name).split() if has_digit(token))


def same_identifier(name: str, alias: str) -> bool:
    """An alias may merge into a name only when both carry the same identifier, or neither carries one."""
    return identifier_tokens(name) == identifier_tokens(alias)
