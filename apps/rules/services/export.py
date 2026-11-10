"""Approved rules as JSON or CSV, with their source and the model that extracted them."""
import csv
import io
import json

from apps.rules.models import Rule

FIELDS = [
    "key", "condition", "action", "exception", "source_quote", "document", "page", "section",
    "confidence", "edited", "reviewer_note", "reviewed_at", "prompt_version", "llm_provider", "llm_model",
]


def approved_rows() -> list[dict]:
    rules = Rule.objects.filter(status=Rule.Status.APPROVED).select_related("document").order_by("key")
    return [
        {
            "key": rule.key,
            "condition": rule.condition,
            "action": rule.action,
            "exception": rule.exception,
            "source_quote": rule.source_quote,
            "document": rule.document.title,
            "page": rule.page,
            "section": rule.section,
            "confidence": rule.confidence,
            "edited": rule.edited,
            "reviewer_note": rule.reviewer_note,
            "reviewed_at": rule.reviewed_at.isoformat() if rule.reviewed_at else "",
            "prompt_version": rule.prompt_version,
            "llm_provider": rule.llm_provider,
            "llm_model": rule.llm_model,
        }
        for rule in rules
    ]


def to_json(rows: list[dict]) -> str:
    return json.dumps({"rules": rows, "count": len(rows)}, indent=2, ensure_ascii=False)


def to_csv(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
