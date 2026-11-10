from django.apps import AppConfig


class KgConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.kg"
    label = "kg"
    verbose_name = "Knowledge graph"

    def ready(self) -> None:
        from apps.kg import mocking  # noqa: F401  registers the offline extractor and answerer
