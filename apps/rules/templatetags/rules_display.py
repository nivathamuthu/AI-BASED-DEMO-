from django import template

from apps.rules.services.present import is_deferral_against_limit

register = template.Library()


@register.filter
def deferral_vs_limit(conflict) -> bool:
    """True when an agent conflict dict is the C-check deferral against a hard limit."""
    if not isinstance(conflict, dict):
        return False
    return is_deferral_against_limit(conflict.get("rule_text", ""), conflict.get("other_text", ""))
