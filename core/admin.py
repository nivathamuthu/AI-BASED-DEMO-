from django.contrib import admin

from .models import EvalRun, LLMCall


@admin.register(EvalRun)
class EvalRunAdmin(admin.ModelAdmin):
    list_display = ("created_at", "kind", "llm_provider", "llm_model", "embeddings_provider", "embeddings_model")
    list_filter = ("kind", "llm_provider", "embeddings_provider")


@admin.register(LLMCall)
class LLMCallAdmin(admin.ModelAdmin):
    list_display = ("created_at", "task", "schema_name", "provider", "attempts", "validation_ok", "latency_ms")
    list_filter = ("provider", "validation_ok", "task")
