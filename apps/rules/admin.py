from django.contrib import admin

from apps.rules.models import Rule, RuleConflict, RuleExtraction, RuleRun


@admin.register(Rule)
class RuleAdmin(admin.ModelAdmin):
    list_display = ("key", "status", "document", "page", "confidence", "prompt_version", "llm_provider", "edited")
    list_filter = ("status", "prompt_version", "llm_provider", "document")
    search_fields = ("key", "condition", "action", "source_quote")
    raw_id_fields = ("chunk", "merged_into")


@admin.register(RuleConflict)
class RuleConflictAdmin(admin.ModelAdmin):
    list_display = ("rule_a", "rule_b", "is_conflict", "kind", "severity", "llm_provider", "created_at")
    list_filter = ("is_conflict", "kind", "severity")


@admin.register(RuleExtraction)
class RuleExtractionAdmin(admin.ModelAdmin):
    list_display = ("chunk", "prompt_version", "kept_count", "llm_provider", "llm_model", "created_at")
    list_filter = ("prompt_version", "llm_provider")
    raw_id_fields = ("chunk",)


@admin.register(RuleRun)
class RuleRunAdmin(admin.ModelAdmin):
    list_display = ("created_at", "prompt_version", "llm_provider", "llm_model", "forced")
