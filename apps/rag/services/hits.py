"""One retrieved chunk, with enough metadata to cite it."""
from dataclasses import dataclass


@dataclass
class Hit:
    chunk_id: int
    text: str
    document_id: int
    document_title: str
    page_start: int | None
    section: str
    score: float
    lexical: int = 0

    def citation(self) -> str:
        parts = [self.document_title]
        if self.page_start:
            parts.append(f"p.{self.page_start}")
        if self.section:
            parts.append(f"§ {self.section}")
        return ", ".join(parts)
