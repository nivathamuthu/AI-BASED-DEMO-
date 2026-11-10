"""Load a versioned Jinja prompt, inject few-shot examples, and render the schema.

Templates live in prompts/<task>/v1.md and v2.md. Few-shot examples live beside
them in examples.yaml. The caller passes the Pydantic class; this module does
not call the model.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml
from django.conf import settings
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel


def prompts_dir() -> Path:
    return Path(settings.BASE_DIR) / "prompts"


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(prompts_dir())),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


def available_versions(task: str) -> list[str]:
    """Versions that have a template file, oldest first."""
    folder = prompts_dir() / task
    return sorted(path.stem for path in folder.glob("v*.md"))


def render_prompt(task: str, version: str, variables: dict, schema: type[BaseModel]) -> str:
    """Render prompts/<task>/<version>.md with examples and the JSON schema."""
    template_name = f"{task}/{version}.md"
    if not (prompts_dir() / template_name).is_file():
        known = ", ".join(available_versions(task)) or "none"
        raise FileNotFoundError(f"No prompt {template_name}. Known versions: {known}")
    payload = dict(variables)
    domain = payload.pop("domain", "aviation maintenance")
    return _environment().get_template(template_name).render(
        domain=domain,
        examples=_example_block(task),
        schema=json.dumps(schema.model_json_schema(), indent=2),
        **payload,
    )


def _example_block(task: str) -> str:
    path = prompts_dir() / task / "examples.yaml"
    if not path.is_file():
        return ""
    rows = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    blocks = []
    for index, row in enumerate(rows, start=1):
        output = json.dumps(row.get("output", {}), ensure_ascii=False)
        blocks.append(
            f"Example {index}\nCHUNK:\n{row.get('chunk', '').strip()}\n"
            f"QUESTION:\n{row.get('question', '').strip()}\nOUTPUT:\n{output}"
        )
    return "\n\n".join(blocks)
