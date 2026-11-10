"""Dispatch to the parser for a detected file type."""
from pathlib import Path

from apps.ingest.services.blocks import Block
from apps.ingest.services.parse_docx import parse_docx
from apps.ingest.services.parse_pdf import parse_pdf
from apps.ingest.services.parse_pptx import parse_pptx
from apps.ingest.services.parse_text import parse_text


def parse_document(path: Path, file_type: str) -> list[Block]:
    """Return the blocks of `path`. `file_type` is one of pdf, docx, pptx, txt, md."""
    if file_type == "pdf":
        return parse_pdf(path)
    if file_type == "docx":
        return parse_docx(path)
    if file_type == "pptx":
        return parse_pptx(path)
    if file_type == "md":
        return parse_text(path, markdown=True)
    if file_type == "txt":
        return parse_text(path, markdown=False)
    raise ValueError(f"No parser for file type {file_type!r}")
