from django.apps import AppConfig


class PromptsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.prompts"
    label = "prompts"
    verbose_name = "Prompts"

    def ready(self) -> None:
        from apps.prompts import mocking  # noqa: F401  (registers mock handlers)
