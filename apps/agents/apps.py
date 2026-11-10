from django.apps import AppConfig


class AgentsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.agents"
    label = "agents"
    verbose_name = "Compliance agent"

    def ready(self) -> None:
        from apps.agents import mocking  # noqa: F401  registers the offline stand-ins for the agent prompts
