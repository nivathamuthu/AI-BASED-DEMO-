"""Plain text and Markdown. Markdown headings (#, ##, ...) and numbered headings become heading blocks."""
import re
from pathlib import Path

from apps.ingest.services.blocks import Block

_MD_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})[.)]\s+(\S.{0,120})$")


def parse_text(path: Path, *, markdown: bool) -> list[Block]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        text = " ".join(paragraph).strip()
        paragraph.clear()
        if text:
            blocks.append(Block(text=text, page=1))

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        heading = _heading(stripped, markdown=markdown)
        if heading is not None:
            flush()
            blocks.append(heading)
            continue
        paragraph.append(stripped)
    flush()
    if not blocks:
        raise ValueError(f"No text extracted from {path.name}")
    return blocks


def _heading(line: str, *, markdown: bool) -> Block | None:
    """A markdown heading, or a short numbered heading such as '1.2 Inspection'."""
    if markdown:
        match = _MD_HEADING.match(line)
        if match and len(line) <= 140:
            return Block(text=match.group(2).strip(), page=1, is_heading=True, heading_level=len(match.group(1)))
    match = _NUMBERED_HEADING.match(line)
    if match and len(line) <= 140:
        level = match.group(1).count(".") + 1
        return Block(text=line, page=1, is_heading=True, heading_level=level)
    return None
