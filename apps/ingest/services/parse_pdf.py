"""PDF to blocks, using pypdf. One block per paragraph; numbered headings stay their own blocks."""
import re
from pathlib import Path

from pypdf import PdfReader

from apps.ingest.services.blocks import Block

# "1. Applicability" or "2.1 Replacement criteria"
_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})[.)]\s+\S.{0,120}$")


def parse_pdf(path: Path) -> list[Block]:
    reader = PdfReader(str(path))
    blocks: list[Block] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        paragraph: list[str] = []

        def flush() -> None:
            text = " ".join(paragraph).strip()
            paragraph.clear()
            if text:
                blocks.append(Block(text=text, page=page_number))

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                flush()
                continue
            if _HEADING.match(line) and len(line) <= 140:
                flush()
                level = line.split()[0].rstrip(".)").count(".") + 1
                blocks.append(Block(text=line, page=page_number, is_heading=True, heading_level=level))
                continue
            paragraph.append(line)
        flush()
    if not blocks:
        raise ValueError(f"No text extracted from {path.name}")
    return blocks
