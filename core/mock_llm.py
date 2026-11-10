"""Deterministic offline stand-in for an LLM (LLM_PROVIDER=mock).

Given a Pydantic schema and the prompt, it returns a dict that validates
against the schema. Modules can register a smarter handler for their own schema
(for example, one that copies a sentence out of the retrieved context) with
`@register(MySchema)`. Everything else falls back to a generic schema walker.
"""
from __future__ import annotations

import enum
import types
import typing
from collections.abc import Callable

import annotated_types as at
from pydantic import BaseModel
from pydantic.fields import FieldInfo

MockHandler = Callable[[str], dict]
_HANDLERS: dict[type[BaseModel], MockHandler] = {}


def register(schema: type[BaseModel]) -> Callable[[MockHandler], MockHandler]:
    """Register a prompt -> dict handler for one schema."""

    def decorator(fn: MockHandler) -> MockHandler:
        _HANDLERS[schema] = fn
        return fn

    return decorator


def mock_response(schema: type[BaseModel], prompt: str) -> dict:
    """Return a schema-valid dict for `schema`, derived deterministically from `prompt`."""
    handler = _HANDLERS.get(schema)
    if handler is not None:
        return handler(prompt)
    return _fill_model(schema, prompt)


def _snippet(prompt: str, limit: int = 120) -> str:
    """A short, deterministic piece of the prompt to use as string content."""
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    text = lines[-1] if lines else "mock"
    return text[:limit]


def _fill_model(schema: type[BaseModel], prompt: str) -> dict:
    data = {}
    for name, field in schema.model_fields.items():
        if not field.is_required():
            continue
        data[name] = _fill_value(field.annotation, field, prompt)
    return data


def _constraint(field: FieldInfo | None, kind: type):
    if field is None:
        return None
    for meta in field.metadata:
        if isinstance(meta, kind):
            return meta
    return None


def _fill_value(annotation, field: FieldInfo | None, prompt: str):
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Annotated:
        return _fill_value(args[0], field, prompt)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        return _fill_value(non_none[0], field, prompt) if non_none else None
    if origin is typing.Literal:
        return args[0]
    if origin in (list, set, tuple, frozenset):
        min_len = _constraint(field, at.MinLen)
        count = max(1, min_len.min_length if min_len else 1)
        return [_fill_value(args[0] if args else str, None, prompt) for _ in range(count)]
    if origin is dict:
        return {}

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return _fill_model(annotation, prompt)
        if issubclass(annotation, enum.Enum):
            return next(iter(annotation)).value
        if annotation is bool:
            return True
        if annotation is int:
            ge = _constraint(field, at.Ge)
            return max(1, int(ge.ge) if ge else 1)
        if annotation is float:
            le = _constraint(field, at.Le)
            return min(0.9, float(le.le)) if le else 0.9
        if annotation is str:
            max_len = _constraint(field, at.MaxLen)
            return _snippet(prompt, max_len.max_length if max_len else 120)
    return None
