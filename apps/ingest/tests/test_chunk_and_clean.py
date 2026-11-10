"""Chunking and header removal. No database."""
from apps.ingest.services.blocks import Block
from apps.ingest.services.chunk import chunk_blocks
from apps.ingest.services.clean import clean_blocks


def test_chunks_follow_headings_and_pages():
    blocks = [
        Block("1. Gear", page=1, is_heading=True, heading_level=1),
        Block("Replace the gear at the next check.", page=1),
        Block("1.1 Bushing", page=2, is_heading=True, heading_level=2),
        Block("Part 65-45210-3 is installed on the side-stay.", page=2),
    ]
    chunks = chunk_blocks(blocks, max_tokens=80, min_tokens=5)

    assert [chunk.section_path for chunk in chunks] == ["1. Gear", "1. Gear > 1.1 Bushing"]
    assert chunks[1].page_start == 2
    assert "65-45210-3" in chunks[1].text
    assert chunks[0].ordinal == 0


def test_long_paragraph_is_split_on_the_token_budget():
    sentence = "The inspection interval is 100 flight hours and the result is recorded. "
    blocks = [
        Block("2. Inspection", page=3, is_heading=True, heading_level=1),
        Block(sentence * 40, page=3),
    ]
    chunks = chunk_blocks(blocks, max_tokens=40, min_tokens=5)

    assert len(chunks) > 1
    assert all(chunk.section_title == "2. Inspection" for chunk in chunks)
    assert all(chunk.token_count <= 80 for chunk in chunks)


def test_repeated_header_is_removed_and_a_one_off_line_is_kept():
    blocks = []
    for page in (1, 2, 3):
        blocks.append(Block("TRAINING EXCERPT", page=page))
        blocks.append(Block(f"Body of page {page} with its own sentence.", page=page))
    blocks.append(Block("A line that appears only once.", page=1))

    cleaned = clean_blocks(blocks)

    assert all(block.text != "TRAINING EXCERPT" for block in cleaned)
    assert any("page 2" in block.text for block in cleaned)
    assert any(block.text == "A line that appears only once." for block in cleaned)
