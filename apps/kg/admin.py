from django.contrib import admin

from apps.kg.models import GraphBuild, KGExtraction


@admin.register(KGExtraction)
class KGExtractionAdmin(admin.ModelAdmin):
    list_display = ("chunk", "prompt_version", "llm_provider", "llm_model", "created_at")
    list_filter = ("prompt_version", "llm_provider")
    raw_id_fields = ("chunk",)


@admin.register(GraphBuild)
class GraphBuildAdmin(admin.ModelAdmin):
    list_display = ("id", "backend", "prompt_version", "node_count", "edge_count",
                    "merged_count", "dropped_count", "created_at")
