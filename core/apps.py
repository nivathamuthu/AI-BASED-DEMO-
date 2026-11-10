import sys

from django.apps import AppConfig
from django.conf import settings


class CoreConfig(AppConfig):
    name = "core"
    verbose_name = "Core (LLM, embeddings, shared)"

    def ready(self) -> None:
        # Tests force the mock in a fixture and must not need the network.
        if "pytest" in sys.modules or not settings.LLM_VALIDATE_ON_STARTUP:
            return
        from core.llm import validate_provider_config

        validate_provider_config()
