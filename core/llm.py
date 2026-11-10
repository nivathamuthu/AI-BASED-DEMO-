"""Thin LLM wrapper. Every LLM output in this project goes through `structured_call`.

    answer = structured_call(AnswerSchema, prompt, task="rag.answer")

Flow per call:
  1. ask the provider for a tool call shaped like the Pydantic schema,
  2. validate the returned arguments with Pydantic,
  3. on a validation/parsing error, send the error back and retry (LLM_MAX_RETRIES),
  4. log the call (attempts, latency, tokens, outcome) to core.LLMCall.

A 429 from the provider is not a validation failure: it is retried with exponential
backoff (LLM_RATE_LIMIT_RETRIES), and at most LLM_MAX_CONCURRENCY calls run at once
across web, worker and management commands (a Redis semaphore).

Providers: anthropic | openai | google | mock (selected by LLM_PROVIDER). API keys are
read by the provider SDKs from ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY;
never from code. A real provider with a missing key or an unknown model stops startup
(`validate_provider_config`); there is no fallback to the mock.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol, TypeVar

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from pydantic import BaseModel, ValidationError

from core import mock_llm
from core.timing import timed

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4.1-mini",
    "google": "gemini-2.5-flash",
    "mock": "mock-structured-v1",
}
API_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}


class LLMOutputError(Exception):
    """The model did not return usable structured output (no tool call, empty, ...)."""


class LLMRateLimitError(Exception):
    """The provider kept answering 429 after every backoff, or a daily quota is used up."""


class LLMValidationError(Exception):
    """All attempts failed validation. `errors` holds one message per attempt."""

    def __init__(self, schema_name: str, errors: list[str]):
        self.errors = errors
        super().__init__(f"{schema_name}: no valid output after {len(errors)} attempt(s): {errors[-1]}")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class Backend(Protocol):
    provider: str
    model: str

    def generate(self, schema: type[BaseModel], messages: list[Message]) -> tuple[dict, Usage]:
        """Return the raw (unvalidated) arguments for `schema` plus token usage."""


@dataclass
class MockBackend:
    provider: str = "mock"
    model: str = DEFAULT_MODELS["mock"]

    def generate(self, schema, messages):
        prompt = "\n".join(m.content for m in messages if m.role == "user")
        return mock_llm.mock_response(schema, prompt), Usage()


@dataclass
class LangChainBackend:
    """Any LangChain chat model that supports tool calling (Anthropic, OpenAI)."""

    provider: str
    model: str
    chat_model: object = field(repr=False)

    def generate(self, schema, messages):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_types = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        lc_messages = [lc_types[m.role](content=m.content) for m in messages]
        runnable = self.chat_model.with_structured_output(
            schema, method="function_calling", include_raw=True
        )
        result = runnable.invoke(lc_messages)
        raw = result["raw"]
        usage = Usage(**{k: (raw.usage_metadata or {}).get(k, 0) for k in ("input_tokens", "output_tokens")})
        if not raw.tool_calls:
            raise LLMOutputError("model returned no tool call")
        return raw.tool_calls[0]["args"], usage


@lru_cache(maxsize=4)
def _langchain_backend(provider: str, model: str, temperature: float) -> LangChainBackend:
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        chat = ChatAnthropic(model=model, temperature=temperature, max_tokens=4096)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(model=model, temperature=temperature, max_retries=0)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # max_retries=0: 429s are retried by structured_call's backoff, so they are logged and counted.
        chat = ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=0)
    else:
        raise ImproperlyConfigured(f"Unknown LLM_PROVIDER {provider!r}; use anthropic, openai, google or mock")
    return LangChainBackend(provider=provider, model=model, chat_model=chat)


def model_name(provider: str | None = None) -> str:
    provider = provider or settings.LLM_PROVIDER
    if provider == "mock":
        return DEFAULT_MODELS["mock"]
    return settings.LLM_MODEL or DEFAULT_MODELS.get(provider, "")


def get_backend() -> Backend:
    """Backend for the configured LLM_PROVIDER."""
    provider = settings.LLM_PROVIDER
    if provider == "mock":
        return MockBackend()
    return _langchain_backend(provider, model_name(provider), settings.LLM_TEMPERATURE)


def validate_provider_config(check_model: bool = True) -> None:
    """Raise ImproperlyConfigured when a real provider cannot work. Called at startup."""
    provider = settings.LLM_PROVIDER
    if provider == "mock":
        return
    if provider not in API_KEY_ENV:
        raise ImproperlyConfigured(f"Unknown LLM_PROVIDER {provider!r}; use anthropic, openai, google or mock")
    key_name = API_KEY_ENV[provider]
    if not os.environ.get(key_name, "").strip():
        raise ImproperlyConfigured(
            f"LLM_PROVIDER={provider} but {key_name} is empty or not passed to this container. "
            f"Set it in .env and recreate the containers (docker compose up -d web worker)."
        )
    if check_model and provider == "google":
        _check_google_model(model_name(provider))


def _check_google_model(name: str) -> None:
    """Ask the Gemini API whether the model exists for this key. Costs no generation quota."""
    from google import genai

    client = genai.Client()
    try:
        info = client.models.get(model=name)
    except Exception as exc:  # the SDK raises several error types; all mean "unusable"
        if is_transient(exc):
            raise ImproperlyConfigured(
                f"Could not reach the Gemini API to check LLM_MODEL={name!r} (network or DNS error): {exc}"
            ) from exc
        raise ImproperlyConfigured(f"Gemini rejected LLM_MODEL={name!r} with this GOOGLE_API_KEY: {exc}") from exc
    finally:
        client.close()
    actions = getattr(info, "supported_actions", None) or []
    if actions and "generateContent" not in actions:
        raise ImproperlyConfigured(f"Gemini model {name!r} does not support generateContent ({actions})")


# --- rate limits ---------------------------------------------------------------

_RETRY_HINT = re.compile(r"retry(?:Delay)?[\"': ]+(?:in )?([\d.]+)\s*s", re.IGNORECASE)


def _error_chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def rate_limit_info(exc: BaseException) -> tuple[bool, bool, float | None]:
    """(is_rate_limit, is_daily_quota, retry_after_seconds) for a provider exception."""
    for item in _error_chain(exc):
        status = getattr(item, "status_code", None) or getattr(item, "code", None)
        text = str(item)
        lowered = text.lower()
        worded = re.search(r"\b429\b", text) and any(w in lowered for w in ("quota", "rate", "exhausted", "too many"))
        if status == 429 or worded or "RESOURCE_EXHAUSTED" in text:
            daily = "PerDay" in text or "per day" in text.lower()
            hint = _RETRY_HINT.search(text)
            return True, daily, float(hint.group(1)) if hint else None
    return False, False, None


_TRANSIENT_NAMES = ("RemoteProtocolError", "ConnectError", "ConnectTimeout", "ReadTimeout", "ReadError",
                    "WriteError", "PoolTimeout", "ServerError", "ServiceUnavailable", "InternalServerError")


def is_transient(exc: BaseException) -> bool:
    """A dropped connection, timeout or 5xx: worth the same backoff as a 429."""
    for item in _error_chain(exc):
        status = getattr(item, "status_code", None) or getattr(item, "code", None)
        if isinstance(status, int) and 500 <= status < 600:
            return True
        if type(item).__name__ in _TRANSIENT_NAMES:
            return True
    return False


def _generate_with_backoff(backend: Backend, schema, messages, task: str):
    """backend.generate inside a concurrency slot, retrying 429s, 5xx and dropped connections with backoff."""
    retries = settings.LLM_RATE_LIMIT_RETRIES
    for attempt in range(retries + 1):
        try:
            with llm_slot(backend.provider):
                return backend.generate(schema, messages)
        except (ValidationError, LLMOutputError):
            raise
        except Exception as exc:
            limited, daily, hint = rate_limit_info(exc)
            if not limited and not is_transient(exc):
                raise
            if daily:
                raise LLMRateLimitError(f"{backend.provider} daily quota exhausted for {backend.model}: {exc}") from exc
            if attempt == retries:
                raise LLMRateLimitError(
                    f"{backend.provider} still failing after {retries} backoffs ({task}): {exc}"
                ) from exc
            wait = min(settings.LLM_BACKOFF_MAX_SECONDS, settings.LLM_BACKOFF_BASE_SECONDS * 2 ** attempt)
            wait = max(wait, hint or 0)
            reason = "429" if limited else type(exc).__name__
            logger.warning("%s: %s from %s, backing off %.1fs (%d/%d)", task, reason, backend.provider, wait,
                           attempt + 1, retries)
            time.sleep(wait)
    raise AssertionError("unreachable")


_local_slots: dict[int, threading.BoundedSemaphore] = {}
_local_lock = threading.Lock()


def _local_semaphore(limit: int) -> threading.BoundedSemaphore:
    with _local_lock:
        return _local_slots.setdefault(limit, threading.BoundedSemaphore(limit))


@lru_cache(maxsize=1)
def _redis_client():
    import redis

    return redis.Redis.from_url(settings.CELERY_BROKER_URL)


@contextmanager
def llm_slot(provider: str):
    """Hold one of LLM_MAX_CONCURRENCY slots shared by every process (Redis), for one provider call."""
    limit = settings.LLM_MAX_CONCURRENCY
    if provider == "mock" or limit <= 0:
        yield
        return
    try:
        import redis

        client = _redis_client()
        client.ping()
    except Exception as exc:
        logger.warning("LLM concurrency: Redis unavailable (%s); limiting this process only", exc)
        with _local_semaphore(limit):
            yield
        return
    deadline = time.monotonic() + settings.LLM_SLOT_WAIT_SECONDS
    while True:
        for index in range(limit):
            lock = client.lock(f"docintel:llm-slot:{index}", timeout=settings.LLM_SLOT_TIMEOUT_SECONDS)
            if lock.acquire(blocking=False):
                try:
                    yield
                finally:
                    try:
                        lock.release()
                    except redis.exceptions.LockError:
                        logger.warning("LLM slot %d expired before release; raise LLM_SLOT_TIMEOUT_SECONDS", index)
                return
        if time.monotonic() > deadline:
            raise LLMRateLimitError(f"No free LLM slot after {settings.LLM_SLOT_WAIT_SECONDS}s (limit {limit})")
        time.sleep(0.25)


def describe_backend() -> dict:
    """Provider and model name for the current LLM, without making a call."""
    provider = settings.LLM_PROVIDER
    return {"provider": provider, "model": model_name(provider)}


def get_chat_model():
    """Raw LangChain chat model, for LangGraph nodes that need tool binding. None in mock mode."""
    backend = get_backend()
    return getattr(backend, "chat_model", None)


def structured_call(
    schema: type[T],
    prompt: str,
    *,
    task: str,
    system: str | None = None,
    prompt_name: str = "",
    prompt_version: str = "",
    max_retries: int | None = None,
    backend: Backend | None = None,
) -> T:
    """Call the LLM and return a validated `schema` instance, retrying on invalid output."""
    from core.models import LLMCall

    backend = backend or get_backend()
    retries = settings.LLM_MAX_RETRIES if max_retries is None else max_retries
    messages = ([Message("system", system)] if system else []) + [Message("user", prompt)]
    errors: list[str] = []
    usage_total = Usage()
    result: T | None = None

    with timed() as timer:
        for attempt in range(1, retries + 2):
            raw: dict | None = None
            try:
                raw, usage = _generate_with_backoff(backend, schema, messages, task)
                usage_total.input_tokens += usage.input_tokens
                usage_total.output_tokens += usage.output_tokens
                result = schema.model_validate(raw)
                break
            except (ValidationError, LLMOutputError) as exc:
                errors.append(str(exc))
                logger.warning("%s attempt %d invalid: %s", task, attempt, exc)
                messages = messages + [
                    Message("assistant", f"(invalid output) {raw}"),
                    Message(
                        "user",
                        "Your previous output failed schema validation:\n"
                        f"{exc}\nReturn a corrected output that satisfies the schema exactly.",
                    ),
                ]

    LLMCall.objects.create(
        task=task,
        schema_name=schema.__name__,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        provider=backend.provider,
        model=backend.model,
        attempts=len(errors) + (1 if result is not None else 0),
        validation_ok=result is not None,
        error=errors[-1] if errors else "",
        latency_ms=timer.duration_ms,
        input_tokens=usage_total.input_tokens,
        output_tokens=usage_total.output_tokens,
    )
    if result is None:
        raise LLMValidationError(schema.__name__, errors)
    return result
