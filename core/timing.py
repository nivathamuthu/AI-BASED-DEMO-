"""Tiny timing helper used for per-stage pipeline timings and LLM latency."""
import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timer:
    started: float = 0.0
    duration_ms: int = 0


@contextmanager
def timed():
    """Measure a block. `duration_ms` is set even if the block raises.

    >>> with timed() as t:
    ...     do_work()
    >>> t.duration_ms
    """
    timer = Timer(started=time.perf_counter())
    try:
        yield timer
    finally:
        timer.duration_ms = int((time.perf_counter() - timer.started) * 1000)
