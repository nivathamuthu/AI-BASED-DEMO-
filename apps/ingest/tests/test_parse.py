"""Parsers against the synthetic corpus builders. No database."""
from scripts.make_synthetic_docs import build_docx, build_pdf, build_pptx
from apps.ingest.services.parse import parse_document


def test_pdf_keeps_pages_headings_and_directive_numbers(tmp_path):
    path = tmp_path / "ad.pdf"
    build_pdf(path)
    blocks = parse_document(path, "pdf")

    text = "\n".join(block.text for block in blocks)
    headings = [block.text for block in blocks if block.is_heading]
    assert "AD 2023-14-09" in text
    assert "PN-77-4418" in text
    assert "65-45210-3" in text
    assert any("Applicability" in heading for heading in headings)
    assert max(block.page for block in blocks) >= 2


def test_docx_uses_heading_styles_and_page_breaks(tmp_path):
    path = tmp_path / "procedure.docx"
    build_docx(path)
    blocks = parse_document(path, "docx")

    headings = [block for block in blocks if block.is_heading]
    text = "\n".join(block.text for block in blocks)
    assert any(block.heading_level == 2 for block in headings)
    assert "MPP-DOCX-4401" in text
    assert max(block.page for block in blocks) >= 2


def test_pptx_makes_each_slide_a_page(tmp_path):
    path = tmp_path / "briefing.pptx"
    build_pptx(path)
    blocks = parse_document(path, "pptx")

    text = "\n".join(block.text for block in blocks)
    assert "MPP-PPTX-4402" in text
    assert "AD 2023-14-09" in text
    assert max(block.page for block in blocks) == 4
    assert sum(1 for block in blocks if block.is_heading) == 4


def test_synthetic_files_are_byte_stable(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        build_pdf(directory / "a.pdf")
        build_docx(directory / "a.docx")
        build_pptx(directory / "a.pptx")
    for name in ("a.pdf", "a.docx", "a.pptx"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
