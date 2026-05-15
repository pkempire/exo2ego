#!/usr/bin/env python3
"""Render front-matter update pages for the final submission PDF."""
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
OUT = ROOT / "paper" / "submission_front_matter.pdf"

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("Title", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22)
SUB = ParagraphStyle("Sub", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=9, leading=11, textColor=colors.HexColor("#555"))
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13.5, leading=16, spaceBefore=8, spaceAfter=5)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.2, leading=12, spaceAfter=6)
SMALL = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#555"))


def safe(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def p(s, st=BODY):
    return Paragraph(safe(s), st)


def fig(name, max_w=6.7 * inch, max_h=5.5 * inch):
    path = FIG / name
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def table(rows, widths=None):
    t = Table(rows, hAlign="LEFT", colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
    ]))
    return t


def main():
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=.72*inch, rightMargin=.72*inch, topMargin=.62*inch, bottomMargin=.62*inch)
    story = [
        p("EgoJudge: Final Submission Update", TITLE),
        p("This front section summarizes the latest own-capture results; the following pages preserve the fuller semester-long project report.", SUB),
        p("Final Approach", H1),
        p("The project is best framed as a perception-conditioned data engine, not a magic single-image reconstruction model. We sync ego/exo captures, generate candidate ego views from the full exo frame, and select candidates using robot-relevant constraints: object identity, hand-object contact, left/right layout, ego-view plausibility, arm continuity, and action phase."),
        fig("final_pipeline_overview.png", max_h=2.9*inch),
        p("Figure A. Final pipeline. Paired ego is used for sync/evaluation, not as input to zero-shot generation.", SMALL),
        p("Latest Own-Capture Sync", H1),
        table([
            ["Event", "Exo", "Ego", "Offset"],
            ["start clap", "01:21.722", "06:08.110", "04:46.388"],
            ["good footage/object sound", "02:06.280", "06:52.688", "04:46.408"],
            ["end clap", "02:17.268", "07:03.655", "04:46.387"],
        ], [1.8*inch, 1.0*inch, 1.0*inch, 1.0*inch]),
        Spacer(1, .08*inch),
        p("Generation uses the full synced window. Paired GT evaluation uses only the good window: exo 02:06.280-02:17.268 / ego 06:52.688-07:03.655. Ego GT frames are rotated 180 degrees for evaluation/display because the raw phone footage is upside down."),
        fig("take2_generated_good_contact_sheet.jpg", max_h=5.0*inch),
        p("Figure B. Good-window demo after fixing GT orientation: exo source, generated ego, flipped ego GT.", SMALL),
        PageBreak(),
        p("Best-of-K Variant Test", H1),
        p("We generated two extra action-phase-conditioned variants on the key move/swap frame. A VLM judge ranked action_phase_01 highest (95/100), base generation second (90/100), and action_phase_02 lower (83/100) because it flipped the ghee/salt layout. This supports the best-of-K + judge framing: generate several plausible ego candidates, then reject the ones with layout or interaction errors."),
        fig("take2_variant_sheet_frame009.jpg", max_h=2.2*inch),
        table([
            ["Candidate", "VLM score", "Main note"],
            ["action_phase_01", "95", "best object/layout/contact preservation"],
            ["base generation", "90", "strong, minor background/book mismatch"],
            ["action_phase_02", "83", "realistic but flips ghee/salt layout"],
            ["flipped ego GT", "65", "real camera but different visible background from exo source"],
        ], [1.5*inch, .8*inch, 4.0*inch]),
        p("Action Phases", H1),
        p("Action phases should be included. They are easy to derive from the same signals we already compute: object masks, hand-object contact, object displacement, and sync markers. For the latest take we export setup/poor-framing, contact_start, lift_or_move, place_release, book_cleanup, and end_reset. This gets closer to the kind of structured robot data that matters: not just a frame, but a state transition."),
        p("Evaluator Reality Check", H1),
        p("We cannot guarantee hand-object interaction correctness from 2D alone. The current reliable checks are SAM2 mask contact/order/layout plus VLM critique. True 3D guarantees require HaMeR/MANO hand meshes or calibrated multi-view depth. HaMeR is wrapped but blocked locally by the gated MANO_RIGHT.pkl asset and environment setup."),
    ]
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
