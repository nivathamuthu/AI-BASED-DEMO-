"""Token counts for chunk sizing.

Uses tiktoken cl100k_base (baked into the image at build time). If that encoding
cannot be loaded, falls back to one token per four characters so the pipeline
still runs with no network.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Approximate token count of `text`. Empty text is 0."""
    if not text:
        return 0
    encoder = _encoder()
    if encoder is None:
        return max(1, (len(text) + 3) // 4)
    return len(encoder.encode(text))
