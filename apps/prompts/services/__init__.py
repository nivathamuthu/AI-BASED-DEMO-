"""Render a versioned prompt and validate the model output."""
from pydantic import BaseModel

from apps.prompts.schemas import SCHEMAS
from core.llm import structured_call
from core.prompts import render_prompt


def run_prompt(task: str, version: str, variables: dict, schema: type[BaseModel] | None = None):
    """Return a validated schema instance. Invalid model output is retried inside structured_call."""
    schema = schema or SCHEMAS[task]
    prompt = render_prompt(task, version, variables, schema)
    return structured_call(
        schema,
        prompt,
        task=f"prompts.{task}",
        prompt_name=task,
        prompt_version=version,
    )
