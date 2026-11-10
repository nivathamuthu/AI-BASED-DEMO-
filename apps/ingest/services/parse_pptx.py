"""PPTX to blocks. Each slide is a page. The slide title is a heading."""
from pathlib import Path

from pptx import Presentation

from apps.ingest.services.blocks import Block


def parse_pptx(path: Path) -> list[Block]:
    presentation = Presentation(str(path))
    blocks: list[Block] = []
    for page, slide in enumerate(presentation.slides, start=1):
        title_shape = slide.shapes.title
        if title_shape is not None and title_shape.has_text_frame:
            title = title_shape.text_frame.text.strip()
            if title:
                blocks.append(Block(text=title, page=page, is_heading=True, heading_level=1))
        for shape in slide.shapes:
            if title_shape is not None and shape.shape_id == title_shape.shape_id:
                continue
            if not getattr(shape, "has_text_frame", False):
                continue
            lines = [p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip()]
            if lines:
                blocks.append(Block(text="\n".join(lines), page=page))
    if not blocks:
        raise ValueError(f"No text extracted from {path.name}")
    return blocks
