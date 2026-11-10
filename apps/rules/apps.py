from django.apps import AppConfig


class RulesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.rules"
    label = "rules"
    verbose_name = "Rules"

    def ready(self) -> None:
        from apps.rules import mocking  # noqa: F401  registers the offline extractor and judge
