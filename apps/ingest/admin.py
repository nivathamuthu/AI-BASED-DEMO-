from django.contrib import admin

from .models import Chunk, Document, PipelineRun, PipelineStage


class StageInline(admin.TabularInline):
    model = PipelineStage
    extra = 0
    readonly_fields = ("name", "status", "attempts", "duration_ms", "error", "summary")


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "file_type", "status", "page_count", "size_bytes", "created_at")
    search_fields = ("title", "original_name", "sha256")
    readonly_fields = ("sha256",)


@admin.register(PipelineRun)
class PipelineRunAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "status", "current_stage", "started_at", "finished_at")
    inlines = [StageInline]


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("document", "ordinal", "page_start", "section_title", "token_count")
    search_fields = ("text", "section_path")
