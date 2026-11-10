from django.contrib import admin

from apps.agents.models import AgentRun, AgentStep


class StepInline(admin.TabularInline):
    model = AgentStep
    extra = 0
    fields = ("order", "node", "loop", "status", "duration_ms", "error")
    readonly_fields = fields


@admin.register(AgentRun)
class AgentRunAdmin(admin.ModelAdmin):
    list_display = ("id", "question", "intent", "status", "loops", "verified", "llm_provider", "created_at")
    list_filter = ("status", "intent", "source")
    inlines = [StepInline]
