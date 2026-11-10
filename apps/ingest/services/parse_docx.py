"""DOCX to blocks. Heading styles become heading blocks. A page break bumps the page number."""
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from apps.ingest.services.blocks import Block


def parse_docx(path: Path) -> list[Block]:
    document = Document(str(path))
    blocks: list[Block] = []
    page = 1
    for item in _body_items(document):
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if text:
                level = _heading_level(item)
                if level:
                    blocks.append(Block(text=text, page=page, is_heading=True, heading_level=level))
                else:
                    blocks.append(Block(text=text, page=page))
            if _has_page_break(item):
                page += 1
        else:
            for row in item.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                # A row repeats the same text when cells are merged. Drop the copies.
                unique = list(dict.fromkeys(cells))
                if unique:
                    blocks.append(Block(text=" | ".join(unique), page=page))
    if not blocks:
        raise ValueError(f"No text extracted from {path.name}")
    return blocks


def _body_items(document):
    """Paragraphs and tables in document order."""
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _heading_level(paragraph: Paragraph) -> int:
    name = paragraph.style.name if paragraph.style is not None else ""
    if not name.startswith("Heading"):
        return 0
    tail = name.rsplit(" ", 1)[-1]
    return int(tail) if tail.isdigit() else 1


def _has_page_break(paragraph: Paragraph) -> bool:
    xml = paragraph._p.xml
    return 'w:type="page"' in xml or "lastRenderedPageBreak" in xml
