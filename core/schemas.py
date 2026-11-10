"""Pydantic types shared across modules."""
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Where a piece of text came from. Rendered as `Title, p.3, §4.2 Deferral`."""

    document_id: str
    document_title: str
    chunk_id: int | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section: str = ""

    def __str__(self) -> str:
        parts = [self.document_title]
        if self.page_start:
            pages = f"p.{self.page_start}"
            if self.page_end and self.page_end != self.page_start:
                pages = f"pp.{self.page_start}-{self.page_end}"
            parts.append(pages)
        if self.section:
            parts.append(f"§ {self.section}")
        return ", ".join(parts)
