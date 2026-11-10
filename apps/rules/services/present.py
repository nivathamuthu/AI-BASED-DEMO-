"""Display labels for stored conflict verdicts.

The judge's row is left as it was saved. This only changes how one pair is shown:
a C-check deferral against a hard replacement limit (R-0005 vs R-0006) is a possible
conflict at medium severity. Two different hard limits, such as 11,500 vs 12,000,
stay high.
"""


def is_deferral_against_limit(text_a: str, text_b: str) -> bool:
    """True when one side defers the job to the next C-check and the other is a hard limit."""
    return _one_each(text_a, text_b, _ccheck, _hard_limit)


def present_conflict(text_a: str, text_b: str, severity: str) -> tuple[str, str]:
    """(severity, label) for a conflict row. Only the deferral-vs-limit pair is relabeled."""
    if severity == "high" and is_deferral_against_limit(text_a, text_b):
        return "medium", "possible conflict"
    return severity, "conflict"


def _one_each(text_a: str, text_b: str, left, right) -> bool:
    return (left(text_a) and right(text_b)) or (left(text_b) and right(text_a))


def _ccheck(text: str) -> bool:
    lowered = (text or "").lower()
    return "c-check" in lowered or "c check" in lowered


def _hard_limit(text: str) -> bool:
    lowered = (text or "").lower().replace(",", "")
    if _ccheck(lowered):
        return False
    return "replac" in lowered and ("12000" in lowered or "11500" in lowered)
