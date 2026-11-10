"""Build the fictional maintenance documents used to demo every module.

The files are deterministic (fixed timestamps, no random ids) so ingesting them
twice hits the file-hash dedupe instead of creating a second copy.

Outputs, under data/corpus/synthetic/:
  airworthiness_directive_excerpt.pdf   part numbers and AD 2023-xx references
  maintenance_planning_procedure.docx
  maintenance_planning_briefing.pptx
  engineering_order_EO-2291.docx        contradicts the procedure's 12000 h limit (rules demo)

Nothing in these files is an operational maintenance instruction.
"""
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "corpus" / "synthetic"
FIXED = datetime(2024, 6, 1, 0, 0, 0)

PDF_NAME = "airworthiness_directive_excerpt.pdf"
DOCX_NAME = "maintenance_planning_procedure.docx"
PPTX_NAME = "maintenance_planning_briefing.pptx"
EO_NAME = "engineering_order_EO-2291.docx"

_GEAR = (
    "The main landing gear side-stay is a life-limited component. "
    "The illustrated part number is 65-45210-3. "
    "A worn bushing is replaced with PN-77-4418, which is bought as a matched set "
    "with the side-stay pin. "
    "Record the part number, the serial number and the flight hours at installation "
    "in the maintenance record before the aircraft is released."
)


def build_pdf(path: Path) -> None:
    """A three-page advisory excerpt. Headings are numbered so the chunker can see them."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    heading = ParagraphStyle("SectionHead", parent=styles["Heading2"], fontName="Times-Bold", fontSize=13, spaceBefore=8)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Times-Roman", fontSize=11, leading=14)

    sections = [
        (
            "1. Applicability",
            "This training excerpt restates, in shortened form, the compliance conditions of "
            "AD 2023-14-09 for the main landing gear side-stay. It applies to the narrow-body "
            "fleet when part number 65-45210-3 is installed in the side-stay assembly, serial "
            "numbers S1000 through S1899. AD 2023-07-02 covers the same airframe where the "
            "earlier bushing PN-77-4410 is still fitted, and it is listed here only so the two "
            "directives can be told apart. " + _GEAR,
        ),
        (
            "2. Required action",
            "Replace the side-stay before the accumulated flight hours on part 65-45210-3 reach "
            "12000. The replacement part is PN-77-4418. Do the replacement at the next C-check "
            "when the hours are already above 11000, so the aircraft is not grounded between "
            "checks. Torque the pin to the value in the component maintenance manual and record "
            "the new part number against AD 2023-14-09. A ferry flight with the part over the "
            "limit is not permitted under this excerpt. " + _GEAR,
        ),
        (
            "2.1 Exception",
            "The replacement may be deferred for 50 flight hours when the bushing PN-77-4418 is "
            "on order and a detailed visual inspection of part 65-45210-3 shows no crack. "
            "The deferral is written up as a maintenance item, not as an MEL deferral, and it "
            "expires at 50 flight hours or at the arrival of the part, whichever comes first. "
            "This exception does not apply if AD 2023-07-02 is already open on the same gear.",
        ),
        (
            "3. Related material",
            "Use AD 2023-14-09 together with the component maintenance manual section for the "
            "side-stay. The superseded directive AD 2023-07-02 remains the reference for any "
            "aircraft that has not yet embodied PN-77-4418. Quote the directive number and the "
            "part number in the work order so a later search can find this page by the exact "
            "string AD 2023-14-09 or the exact string PN-77-4418.",
        ),
    ]

    story = []
    for title, text in sections:
        story.append(Paragraph(title, heading))
        story.append(Spacer(1, 6))
        story.append(Paragraph(text, body))
        story.append(PageBreak())

    class StableCanvas(Canvas):
        def __init__(self, *args, **kwargs):
            kwargs["invariant"] = 1
            kwargs["pageCompression"] = 0
            super().__init__(*args, **kwargs)

    def draw_header(canvas, doc):
        canvas.setFont("Times-Roman", 8)
        canvas.drawString(72, 760, "TRAINING EXCERPT - NOT FOR OPERATIONAL USE")

    path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(path), pagesize=letter, title="AD 2023-14-09 training excerpt", author="DocIntel Lab")
    document.build(story, onFirstPage=draw_header, onLaterPages=draw_header, canvasmaker=StableCanvas)


def build_docx(path: Path) -> None:
    """A maintenance planning procedure with Heading styles and one page break."""
    from docx import Document
    from docx.enum.text import WD_BREAK

    document = Document()
    props = document.core_properties
    props.author = "DocIntel Lab"
    props.last_modified_by = "DocIntel Lab"
    props.title = "Maintenance Planning Procedure"
    props.created = FIXED
    props.modified = FIXED
    props.revision = 1

    document.add_heading("Maintenance Planning Procedure", level=0)
    document.add_paragraph(
        "Training document MPP-DOCX-4401. Not an operational procedure. "
        "It describes how the planning office turns a directive into a scheduled task."
    )
    document.add_heading("1. Purpose", level=1)
    document.add_paragraph(
        "Give the planning office one rule for the life-limited side-stay so the same "
        "decision is made on every aircraft. The rule is tied to AD 2023-14-09 and to "
        "part number 65-45210-3."
    )
    document.add_heading("2. Life-limited components", level=1)
    document.add_paragraph(_GEAR)
    document.add_heading("2.1 Replacement criteria", level=2)
    document.add_paragraph(
        "IF part 65-45210-3 is installed AND its flight hours are at least 12000, "
        "THEN replace it with PN-77-4418 before the next revenue flight. "
        "Exception: the part may stay fitted for 50 flight hours when PN-77-4418 is on "
        "order and the inspection in section 2.2 finds no crack. "
        "Procedure code MPP-DOCX-4401."
    )
    document.add_heading("2.2 Inspection", level=2)
    document.add_paragraph(
        "Do a detailed visual inspection of the side-stay at every C-check. "
        "Record the flight hours, the part number and the directive AD 2023-14-09 "
        "on the work card. A crack of any length closes the exception in section 2.1."
    )
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    document.add_heading("3. Deferral", level=1)
    document.add_paragraph(
        "A missing non-essential placard may be deferred under the MEL. "
        "The side-stay replacement itself cannot be deferred under the MEL. "
        "Use the exception in section 2.1, or ground the aircraft. "
        "This is the sentence a later demo uses to separate an MEL deferral from a life limit."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    document.save(buffer)
    path.write_bytes(_stable_zip(buffer.getvalue()))


def build_pptx(path: Path) -> None:
    """A four-slide briefing. Each slide title becomes a section heading."""
    from pptx import Presentation

    presentation = Presentation()
    props = presentation.core_properties
    props.author = "DocIntel Lab"
    props.last_modified_by = "DocIntel Lab"
    props.title = "Maintenance Planning Briefing"
    props.created = FIXED
    props.modified = FIXED
    props.revision = 1

    slides = [
        (
            "Maintenance planning briefing",
            [
                "Training briefing MPP-PPTX-4402. Not for operational use.",
                "Subject: life limit on the main landing gear side-stay.",
            ],
        ),
        (
            "Intervals",
            [
                "Replace part 65-45210-3 at 12000 flight hours.",
                "Start planning the job at 11000 flight hours, at the next C-check.",
                "A 50 flight hour exception applies only while the replacement is on order.",
            ],
        ),
        (
            "Parts",
            [
                "Installed part 65-45210-3, serials S1000 through S1899.",
                "Replacement bushing P/N 77-4418, issued as a matched set with the pin.",
                "Briefing code MPP-PPTX-4402.",
            ],
        ),
        (
            "Directives",
            [
                "Governing directive: AD 2023-14-09.",
                "Earlier directive, for the old bushing only: AD 2023-07-02.",
                "Quote AD 2023-14-09 and PN-77-4418 together on the work order.",
            ],
        ),
    ]
    for title, lines in slides:
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = title
        frame = slide.placeholders[1].text_frame
        frame.text = lines[0]
        for line in lines[1:]:
            frame.add_paragraph().text = line

    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    presentation.save(buffer)
    path.write_bytes(_stable_zip(buffer.getvalue()))


def _stable_zip(data: bytes) -> bytes:
    """Rewrite an Office zip with fixed entry timestamps, so the same content has the same sha256."""
    import zipfile

    source = zipfile.ZipFile(BytesIO(data))
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            entry = zipfile.ZipInfo(info.filename, date_time=FIXED.timetuple()[:6])
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            target.writestr(entry, source.read(info.filename))
    return output.getvalue()


def build_engineering_order(path: Path) -> None:
    """A one-page order that contradicts the procedure's 12000 h limit, for the rules conflict demo."""
    from docx import Document

    document = Document()
    props = document.core_properties
    props.author = "DocIntel Lab"
    props.last_modified_by = "DocIntel Lab"
    props.title = "Engineering Order EO-2291"
    props.created = FIXED
    props.modified = FIXED
    props.revision = 1

    document.add_heading("Engineering Order EO-2291", level=0)
    document.add_paragraph(
        "Training document EO-2291. Not an operational instruction. "
        "It is deliberately inconsistent with MPP-DOCX-4401 so the rules review has a contradiction to show."
    )
    document.add_heading("1. Side-stay limit", level=1)
    document.add_paragraph(
        "IF part 65-45210-3 is installed on the narrow-body fleet, "
        "THEN replace it at 11500 flight hours."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    document.save(buffer)
    path.write_bytes(_stable_zip(buffer.getvalue()))


def build_all(directory: Path = OUT_DIR) -> list[Path]:
    """Write all four files and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / PDF_NAME, directory / DOCX_NAME, directory / PPTX_NAME, directory / EO_NAME]
    build_pdf(paths[0])
    build_docx(paths[1])
    build_pptx(paths[2])
    build_engineering_order(paths[3])
    return paths


def main() -> None:
    paths = build_all()
    print(f"Wrote {len(paths)} files to {OUT_DIR}")
    for path in paths:
        print(f"  {path.name:48} {path.stat().st_size:7} bytes")


if __name__ == "__main__":
    sys.exit(main())
