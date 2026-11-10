"""Structure-aware chunking.

Headings open a section and are kept in the chunk text, so a retrieved chunk
still says which section it came from. A chunk closes when the next heading
starts or when adding the next paragraph would pass the token budget. A paragraph
longer than the budget is split on sentence boundaries.
"""
import re
from dataclasses import asdict, dataclass

from apps.ingest.services.blocks import Block
from apps.ingest.services.tokens import count_tokens

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkDraft:
    ordinal: int
    text: str
    page_start: int | None
    page_end: int | None
    section_title: str
    section_path: str
    heading_level: int | None
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkDraft":
        return cls(**data)


def chunk_blocks(blocks: list[Block], max_tokens: int = 300, min_tokens: int = 40) -> list[ChunkDraft]:
    """Group `blocks` into chunks of at most `max_tokens` tokens, breaking on headings."""
    if max_tokens < 20:
        raise ValueError("max_tokens must be at least 20")
    section: list[tuple[int, str]] = []
    buffer: list[Block] = []
    buffer_tokens = 0
    chunks: list[ChunkDraft] = []

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        text = "\n\n".join(block.text for block in buffer).strip()
        pages = [block.page for block in buffer if block.page]
        chunks.append(
            ChunkDraft(
                ordinal=len(chunks),
                text=text,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                section_title=section[-1][1] if section else "",
                section_path=" > ".join(title for _, title in section),
                heading_level=section[-1][0] if section else None,
                token_count=count_tokens(text),
            )
        )
        buffer = []
        buffer_tokens = 0

    def push(block: Block) -> None:
        nonlocal buffer_tokens
        buffer.append(block)
        buffer_tokens += count_tokens(block.text)

    for block in blocks:
        if block.is_heading:
            flush()
            level = block.heading_level or 1
            while section and section[-1][0] >= level:
                section.pop()
            section.append((level, block.text))
            push(block)
            continue
        for piece in _pieces(block, max_tokens):
            piece_tokens = count_tokens(piece.text)
            if buffer and buffer_tokens + piece_tokens > max_tokens:
                # Keep a heading with the first paragraph of its section, even if that
                # runs a few tokens over the budget. A heading alone is a poor chunk.
                heading_only = len(buffer) == 1 and buffer[0].is_heading
                if not heading_only:
                    flush()
            push(piece)
            if buffer_tokens > max_tokens and not (len(buffer) == 1 and buffer[0].is_heading):
                flush()
    flush()
    _merge_tiny_tail(chunks, min_tokens)
    for index, chunk in enumerate(chunks):
        chunk.ordinal = index
    if not chunks:
        raise ValueError("Chunking produced nothing")
    return chunks


def _pieces(block: Block, max_tokens: int) -> list[Block]:
    """The block itself, or sentence-sized pieces when it is over the budget."""
    if count_tokens(block.text) <= max_tokens:
        return [block]
    pieces: list[Block] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in _sentences(block.text, max_tokens):
        sentence_tokens = count_tokens(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            pieces.append(Block(" ".join(current), page=block.page))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        pieces.append(Block(" ".join(current), page=block.page))
    return pieces


def _sentences(text: str, max_tokens: int) -> list[str]:
    parts = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    output: list[str] = []
    for part in parts or [text]:
        if count_tokens(part) <= max_tokens:
            output.append(part)
            continue
        line: list[str] = []
        for word in part.split():
            candidate = " ".join(line + [word])
            if line and count_tokens(candidate) > max_tokens:
                output.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            output.append(" ".join(line))
    return output or [text]


def _merge_tiny_tail(chunks: list[ChunkDraft], min_tokens: int) -> None:
    """A trailing stub of the same section is more useful glued to the previous chunk."""
    if len(chunks) < 2 or chunks[-1].token_count >= min_tokens:
        return
    previous, last = chunks[-2], chunks[-1]
    if previous.section_path != last.section_path:
        return
    previous.text = previous.text + "\n\n" + last.text
    previous.token_count = count_tokens(previous.text)
    previous.page_end = last.page_end or previous.page_end
    chunks.pop()
