"""JSON files passed from one Celery stage to the next.

The tasks are separate processes, so parsed text cannot stay in memory.
"""
import json
from pathlib import Path

from django.conf import settings

from apps.ingest.services.blocks import Block


def directory(document_id: int) -> Path:
    path = Path(settings.MEDIA_ROOT) / "pipeline" / str(document_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_blocks(document_id: int, name: str, blocks: list[Block]) -> int:
    path = directory(document_id) / f"{name}.json"
    path.write_text(json.dumps([block.to_dict() for block in blocks]), encoding="utf-8")
    return len(blocks)


def load_blocks(document_id: int, name: str) -> list[Block]:
    path = directory(document_id) / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Block.from_dict(item) for item in data]


def save_json(document_id: int, name: str, payload) -> None:
    path = directory(document_id) / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_json(document_id: int, name: str):
    path = directory(document_id) / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))
