import json
import os

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from pydantic import BaseModel, Field

from core.embeddings import describe as describe_embeddings
from core.embeddings import embed_texts
from core.llm import API_KEY_ENV, describe_backend, structured_call, validate_provider_config
from core.models import LLMCall
from core.timing import timed

SENTENCE = "Replace the side-stay before the accumulated flight hours on part 65-45210-3 reach 12000."


class LimitCheck(BaseModel):
    part_number: str = Field(description="The part number in the sentence, exactly as written.")
    flight_hours: int = Field(ge=1, description="The flight-hour limit as an integer.")
    action: str = Field(min_length=1, max_length=80, description="What must be done, in a few words.")


class Command(BaseCommand):
    help = "Make one real structured LLM call and print provider, model, latency and the validated output."

    def add_arguments(self, parser):
        parser.add_argument("--embeddings", action="store_true", help="Also embed one sentence with EMBEDDINGS_PROVIDER.")

    def handle(self, *args, **options):
        backend = describe_backend()
        key_name = API_KEY_ENV.get(backend["provider"])
        self.stdout.write(f"provider {backend['provider']} · model {backend['model']} · "
                          f"{key_name or 'no key needed'}{' present' if key_name and os.environ.get(key_name) else ''}"
                          f" · max concurrency {settings.LLM_MAX_CONCURRENCY}")
        if backend["provider"] == "mock":
            self.stdout.write(self.style.WARNING("LLM_PROVIDER=mock: this call goes to the offline stand-in, not a model."))
        validate_provider_config()
        prompt = (
            "Extract the part number, the flight-hour limit and the required action from this maintenance "
            f"sentence.\n\nSENTENCE:\n{SENTENCE}"
        )
        with timed() as timer:
            try:
                result = structured_call(LimitCheck, prompt, task="core.check_llm", prompt_name="check_llm")
            except Exception as exc:
                raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        call = LLMCall.objects.filter(task="core.check_llm").order_by("-id").first()
        self.stdout.write(f"latency {timer.duration_ms} ms · attempts {call.attempts} · "
                          f"tokens in {call.input_tokens} / out {call.output_tokens}")
        self.stdout.write(json.dumps(result.model_dump(), indent=2))
        if result.part_number != "65-45210-3" or result.flight_hours != 12000:
            self.stdout.write(self.style.WARNING("Output validated but does not match the sentence."))
        else:
            self.stdout.write(self.style.SUCCESS("Validated output matches the sentence."))
        if options["embeddings"]:
            info = describe_embeddings()
            with timed() as embed_timer:
                vector = embed_texts([SENTENCE])[0]
            self.stdout.write(f"embeddings {info['provider']} / {info['model']} · {vector.shape[0]}-dim · "
                              f"{embed_timer.duration_ms} ms")
