"""File type detection. The extension is authoritative for Office files, because
they are zip archives and a magic-byte check reports `zip`. PDFs are checked both ways.
"""
from pathlib import Path

import filetype

SUPPORTED = {"pdf", "docx", "pptx", "txt", "md"}


class UnsupportedFileType(Exception):
    """The upload is not a pdf, docx, pptx, txt or md file."""


def detect_file_type(path: Path) -> str:
    """Return one of pdf, docx, pptx, txt, md. Raise UnsupportedFileType otherwise."""
    ext = path.suffix.lower().lstrip(".")
    if ext == "text":
        ext = "txt"
    if ext in SUPPORTED:
        if ext == "pdf":
            kind = filetype.guess(str(path))
            if kind is not None and kind.extension not in {"pdf"}:
                raise UnsupportedFileType(f"{path.name} is named .pdf but its contents are {kind.extension}")
        return ext
    kind = filetype.guess(str(path))
    if kind is not None and kind.extension == "pdf":
        return "pdf"
    raise UnsupportedFileType(
        f"Cannot ingest {path.name}. Supported types: {', '.join(sorted(SUPPORTED))}."
    )
