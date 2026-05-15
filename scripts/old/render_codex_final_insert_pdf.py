#!/usr/bin/env python3
"""Render the Codex final report insert as a clearly named PDF.

This does not touch paper/report.pdf or any older report PDF.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
OUT = ROOT / "paper" / "codex_final_report_insert_do_not_replace_old_report.pdf"

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("Title", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22)
SUB = ParagraphStyle("Sub", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=9, leading=11, textColor=colors.HexColor("#555555"))
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13.5, leading=16, spaceBefore=8, spaceAfter=5)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.4, leading=12.2, spaceAfter=6)
SMALL = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#555555"))


def safe(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def p(text: str, style=BODY) -> Paragraph:
    return Paragraph(safe(text), style)


def fig(name: str, max_w: float = 6.8 * inch, max_h: float = 4.8 * inch) -> Image:
    path = FIG / name
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def table(rows: list[list[str]], col_widths: list[float]) -> Table:
    t = Table(rows, hAlign="LEFT", colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ]
        )
    )
    return t


def main() -> None:
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.62 * inch,
    )
    story = [
        p("Codex Final Report Insert: Exo2Ego / EgoJudge", TITLE),
        p("Additive update only. This PDF is intentionally named so it cannot be confused with the preserved old report.", SUB),
        p("Final Approach", H1),
        p(
            "The final system is a generate-then-judge pipeline. It uses exocentric perception to preserve "
            "robot-relevant facts, generates multiple plausible egocentric candidates, and rejects candidates "
            "that violate object identity, hand attachment, contact, action phase, or explicit layout constraints."
        ),
        fig("final_pipeline_overview.png", max_h=2.7 * inch),
        p("Figure 1. Final pipeline overview. Paired ego is used for sync and evaluation; it is not required as input to zero-shot generation.", SMALL),
        p("Step-by-Step Pipeline", H1),
        p(
            "1. Sync ego/exo video using clap/object-sound anchors. For take2, the good eval window is "
            "exo 02:06.280-02:17.268 and ego 06:52.688-07:03.655. The extracted ego GT frames were upside "
            "down, so final sheets use the corrected 180-degree-flipped frames."
        ),
        p(
            "2. Parse the exo frame into hands, manipulated objects, workspace, and spatial relations using "
            "VLM scene graphs plus Grounded-SAM/SAM2-style masks. MediaPipe 2D landmarks remain a fallback; "
            "HaMeR is the intended 3D hand mesh path but is blocked locally by MANO/dependency setup."
        ),
        p(
            "3. Run monocular-depth point-cloud reprojection as a diagnostic. It back-projects the exo image, "
            "guesses a head-mounted ego camera, and reprojects visible points. On frame 009 it fills only 34.1% "
            "of the ego canvas, which explains why pure reprojection is not enough."
        ),
        fig("take2_pointcloud_reprojection_demo.png", max_h=3.8 * inch),
        p("Figure 2. Point-cloud diagnostic: useful evidence, but far too sparse to be the final condition by itself.", SMALL),
        PageBreak(),
        p("Generation and Variants", H1),
        p(
            "4. Build an action-phase prompt. The final prompt says the frame is in the move/swap phase and "
            "requires the ghee jar, salt container, black/orange book, table, attached hands, hand-object contact, "
            "and raw room lighting. The prompt now explicitly states that the ghee jar must be to the right of the salt."
        ),
        p(
            "5. Generate multiple ego candidates for the same frame. This best-of-K step matters because one "
            "candidate can have better arm continuity while another has the correct object order."
        ),
        fig("take2_variant_sheet_frame009.jpg", max_h=2.65 * inch),
        p("Figure 3. Frame 009 variants. The chosen candidate is selected by exact constraints, not just visual realism.", SMALL),
        p("EgoJudge Eval", H1),
        p(
            "6. Score candidates with EgoJudge. Pixel metrics against flipped GT are reported only as sanity "
            "checks because the real ego crop, tilt, exposure, and head motion differ. The main score is "
            "constraint-level: object presence, hand/contact, action phase, layout, and VLM evidence."
        ),
        table(
            [
                ["candidate", "pass", "fail", "score", "reason"],
                ["base", "11", "2", "0.846", "fails ghee-right-of-salt and book persistence"],
                ["action_phase_01", "12", "1", "0.846", "fails ghee-right-of-salt"],
                ["action_phase_02", "13", "0", "1.000", "passes exact layout/contact/POV checks"],
            ],
            [1.25 * inch, 0.55 * inch, 0.55 * inch, 0.65 * inch, 3.9 * inch],
        ),
        Spacer(1, 0.08 * inch),
        p(
            "This is the key final result: the generic VLM judge liked a realistic candidate, but the explicit "
            "constraint judge chose a different candidate because it preserved the exact physical facts. That is "
            "the right standard for data meant to train robot policies."
        ),
        fig("eval_suite_overview.png", max_h=3.2 * inch),
        p("Figure 4. EgoJudge eval suite: paired sanity checks plus mask/contact/depth/layout and explicit VLM constraints.", SMALL),
        fig("take2_generated_good_contact_sheet.jpg", max_h=3.4 * inch),
        p("Figure 5. Good-window sequence: exo source, generated ego, and corrected real ego GT.", SMALL),
        p("Honest Limitations", H1),
        p(
            "The current pipeline still does not guarantee exact 3D hand-object physics. A stronger version should "
            "wire in HaMeR/MANO hand meshes, object-level pose/depth, denser mask/depth conditions, and a learned "
            "preference model trained from accepted/rejected EgoJudge outputs."
        ),
    ]
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
