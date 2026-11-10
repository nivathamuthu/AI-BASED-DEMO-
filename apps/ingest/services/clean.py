"""Normalize whitespace and drop running headers and footers.

A short line that shows up on three or more pages is treated as a running header
or footer (page numbers included) and removed. Headings are kept.
"""
import re

from apps.ingest.services.blocks import Block

_SPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_blocks(blocks: list[Block]) -> list[Block]:
    normalized = []
    for block in blocks:
        text = _normalize(block.text)
        if text:
            normalized.append(Block(text, block.page, block.is_heading, block.heading_level))
    repeated = _repeated_lines(normalized)
    cleaned: list[Block] = []
    for block in normalized:
        if block.is_heading or block.text not in repeated:
            cleaned.append(block)
    if not cleaned:
        raise ValueError("Cleaning removed every block")
    return cleaned


def _normalize(text: str) -> str:
    text = text.replace("\x0c", "\n").replace("\u00a0", " ")
    text = _SPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _repeated_lines(blocks: list[Block]) -> set[str]:
    pages_by_text: dict[str, set[int]] = {}
    for block in blocks:
        if block.is_heading or block.page is None or len(block.text) > 100:
            continue
        pages_by_text.setdefault(block.text, set()).add(block.page)
    return {text for text, pages in pages_by_text.items() if len(pages) >= 3}
