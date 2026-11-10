"""Provider config checks, 429 backoff and the concurrency slot. Offline: no provider is called."""
import pytest
from django.core.exceptions import ImproperlyConfigured
from pydantic import BaseModel

from core import llm
from core.llm import (
    LLMRateLimitError,
    Usage,
    describe_backend,
    rate_limit_info,
    structured_call,
    validate_provider_config,
)
from core.models import LLMCall


class Echo(BaseModel):
    text: str


class _Quota(Exception):
    status_code = 429


def test_google_without_key_fails_loudly(settings, monkeypatch):
    settings.LLM_PROVIDER = "google"
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ImproperlyConfigured, match="GOOGLE_API_KEY is empty"):
        validate_provider_config()


def test_unknown_provider_fails_loudly(settings):
    settings.LLM_PROVIDER = "gemini"
    with pytest.raises(ImproperlyConfigured, match="Unknown LLM_PROVIDER"):
        validate_provider_config()


def test_rejected_model_fails_loudly(settings, monkeypatch):
    settings.LLM_PROVIDER = "google"
    settings.LLM_MODEL = "gemini-does-not-exist"
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    def reject(name):
        raise ImproperlyConfigured(f"Gemini rejected LLM_MODEL={name!r}")

    monkeypatch.setattr(llm, "_check_google_model", reject)
    with pytest.raises(ImproperlyConfigured, match="gemini-does-not-exist"):
        validate_provider_config()


def test_describe_reports_the_configured_provider_not_the_mock(settings):
    settings.LLM_PROVIDER = "google"
    settings.LLM_MODEL = "gemini-2.5-flash"
    assert describe_backend() == {"provider": "google", "model": "gemini-2.5-flash"}
    settings.LLM_PROVIDER = "mock"
    assert describe_backend()["model"] == "mock-structured-v1"


def test_rate_limit_detection():
    text = "429 RESOURCE_EXHAUSTED. Please retry in 17.5s. {'retryDelay': '17s'}"
    assert rate_limit_info(RuntimeError(text)) == (True, False, 17.5)
    daily = RuntimeError("429 quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    assert rate_limit_info(daily)[:2] == (True, True)
    assert rate_limit_info(_Quota("slow down"))[0] is True
    assert rate_limit_info(ValueError("chunk mentions part 429 of the manual"))[0] is False


@pytest.mark.django_db
def test_429_is_retried_with_backoff_and_counted_once(settings, monkeypatch):
    settings.LLM_RATE_LIMIT_RETRIES = 3
    settings.LLM_BACKOFF_BASE_SECONDS = 1.0
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    class Limited:
        provider = "mock"
        model = "limited"
        calls = 0

        def generate(self, schema, messages):
            self.calls += 1
            if self.calls < 3:
                raise _Quota("429 Too Many Requests")
            return {"text": "ok"}, Usage()

    backend = Limited()
    assert structured_call(Echo, "x", task="test.backoff", backend=backend).text == "ok"
    assert sleeps == [1.0, 2.0]
    assert LLMCall.objects.get(task="test.backoff").attempts == 1


def test_dropped_connections_and_5xx_are_transient():
    class RemoteProtocolError(Exception):
        pass

    class Unavailable(Exception):
        status_code = 503

    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = RemoteProtocolError("Server disconnected without sending a response.")
    assert llm.is_transient(wrapped) is True
    assert llm.is_transient(Unavailable()) is True
    assert llm.is_transient(ValueError("bad schema")) is False


@pytest.mark.django_db
def test_daily_quota_is_not_retried(settings, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: pytest.fail("slept on a daily quota"))

    class Exhausted:
        provider = "mock"
        model = "exhausted"

        def generate(self, schema, messages):
            raise _Quota("429 quota GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(LLMRateLimitError, match="daily quota"):
        structured_call(Echo, "x", task="test.daily", backend=Exhausted())


def test_slot_limits_concurrency_without_redis(settings, monkeypatch):
    settings.LLM_MAX_CONCURRENCY = 1

    def no_redis():
        raise ConnectionError("no redis in this test")

    monkeypatch.setattr(llm, "_redis_client", no_redis)
    semaphore = llm._local_semaphore(1)
    with llm.llm_slot("google"):
        assert semaphore.acquire(blocking=False) is False
    assert semaphore.acquire(blocking=False) is True
    semaphore.release()
