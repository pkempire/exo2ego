#!/usr/bin/env python3
"""Render the cleaned final report PDF without requiring a TeX install."""
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
OUT = ROOT / "paper" / "report.pdf"


def safe(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(safe(text), style)


def fig(name: str, max_w: float = 6.7 * inch, max_h: float = 7.8 * inch):
    path = FIG / name
    if not path.exists():
        return p(f"[missing figure: {name}]", STYLES["Body"])
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def table(rows, widths=None):
    t = Table(rows, hAlign="LEFT", colWidths=widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def bullets(items: list[str], style):
    out = []
    for item in items:
        out.append(Paragraph("&#8226; " + safe(item), style))
    return out


styles = getSampleStyleSheet()
STYLES = {
    "Title": ParagraphStyle("Title", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=8),
    "Sub": ParagraphStyle("Sub", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=9.5, leading=12, textColor=colors.HexColor("#444444"), spaceAfter=14),
    "H1": ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, leading=17, spaceBefore=10, spaceAfter=5),
    "H2": ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, leading=13, spaceBefore=7, spaceAfter=3),
    "Body": ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.4, leading=12.4, spaceAfter=6),
    "Small": ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8.0, leading=10, textColor=colors.HexColor("#555555"), spaceAfter=7),
}


def main() -> None:
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.62 * inch,
        title="EgoJudge Report",
    )

    story = []
    story += [
        p("EgoJudge: Perception-Conditioned Exo-to-Ego Generation for Robot-Learning Data", STYLES["Title"]),
        p("CMSC498E Robotics Final Project - Parth Kocheta - Working Draft", STYLES["Sub"]),
        p(
            "Robot policies need first-person manipulation data: hands, objects, contact, and action phase from the camera viewpoint a robot will eventually use. Third-person human video is easier to collect, but converting it into egocentric video is underconstrained from a single camera. EgoJudge is a generate-then-judge pipeline: extract manipulation evidence from the exo frame, generate ego candidates, and reject candidates that violate robot-relevant constraints.",
            STYLES["Body"],
        ),
        p("One-Sentence Thesis", STYLES["H1"]),
        p("Single-view exo-to-ego is too ambiguous to solve by geometry alone, so the practical path is to use perception to constrain generation and then reject candidates that violate manipulation physics.", STYLES["Body"]),
        p("Final Approach", STYLES["H1"]),
        fig("final_pipeline_overview.png", max_h=3.0 * inch),
        p("Figure 1. EgoJudge pipeline: perception, generation, judging, and paired evaluation when ground truth exists.", STYLES["Small"]),
        *bullets(
            [
                "Sync and select usable frames with audio transients and manual verification.",
                "Extract perception from the exo frame using open-vocabulary detection and SAM2 masks.",
                "Generate candidates from the full exo frame with a structured prompt.",
                "Judge candidates by role presence, contact, spatial order, ego layout, and arm continuity.",
                "Add action-phase labels so the data describes state transitions, not just static images.",
            ],
            STYLES["Body"],
        ),
        p("Why Geometry Alone Failed", STYLES["H1"]),
        p("The early geometry pipeline estimated monocular depth, backprojected a point cloud, guessed a virtual ego camera, and reprojected sparse RGB. It failed for a principled reason: with one uncalibrated exo camera, the ego camera pose is not observable. Multi-view methods and calibrated rigs change that; a single phone camera does not.", STYLES["Body"]),
        fig("geometry_failure.png", max_h=3.2 * inch),
        p("Figure 2. Geometry-only exo-to-ego needs either multi-view calibration or a learned ego-pose prior.", STYLES["Small"]),
        PageBreak(),
        p("Own Capture: Clean Sync", STYLES["H1"]),
        p("The newest usable capture pair is C0008.MP4 as exo and IMG_4530.MOV as ego. Three refined audio markers agree within about 0.02 seconds, giving a stable offset of roughly 04:46.388 from exo to ego.", STYLES["Body"]),
        table(
            [
                ["Event", "Exo time", "Ego time", "Ego - exo"],
                ["start clap", "01:21.722", "06:08.110", "04:46.388"],
                ["good footage / object sound", "02:06.280", "06:52.688", "04:46.408"],
                ["end clap", "02:17.268", "07:03.655", "04:46.387"],
            ],
            [1.9 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch],
        ),
        Spacer(1, 0.08 * inch),
        p("Generate on the full synced window: exo 01:21.722-02:17.268. Evaluate against paired ego only on the good window: exo 02:06.280-02:17.268, ego 06:52.688-07:03.655.", STYLES["Body"]),
        fig("take2_good_sync_pairs.jpg", max_h=6.3 * inch),
        p("Figure 3. Good evaluation window: exo frame on the left, synced full-frame ego target on the right.", STYLES["Small"]),
        PageBreak(),
        p("Generated Demo Frames", STYLES["H1"]),
        p("The full synced segment was generated from exo frames. The sheet below shows the good tail only: exo source, generated ego candidate, and synced ego ground truth. The generated frames are often cleaner and more centered than the real ego camera, which is useful for demo quality but should be evaluated against physical constraints rather than PSNR alone.", STYLES["Body"]),
        fig("take2_generated_good_contact_sheet.jpg", max_h=7.4 * inch),
        p("Figure 4. Good-window generations. These are strong enough for a demo, while the GT column reminds us that the actual ego camera was still imperfect.", STYLES["Small"]),
        PageBreak(),
        p("SAM2 Mask Result And Failure Case", STYLES["H1"]),
        fig("own_capture_sam2_source_overlay.png", max_h=4.4 * inch),
        p("Figure 5. GroundingDINO+SAM2 masks recover hands, table, and book objects on the earlier attached frame.", STYLES["Small"]),
        p("A mask-composed candidate preserved object contact but produced a detached hand. That is not a success; it is exactly why the judge needs arm-continuity checks. A contact score can pass while the image is still physically impossible.", STYLES["Body"]),
        fig("rejected_floating_hand_mask_condition.png", max_h=3.0 * inch),
        p("Figure 6. Rejected floating-hand candidate. This should be used in the report as a failure mode, not a headline result.", STYLES["Small"]),
        p("Action Phases", STYLES["H1"]),
        p("Yes: action phases should be part of our final approach. They are cheap to generate from masks, object displacement, and contact changes, and they make the output closer to the kind of structured robot training data companies actually buy. For the newest take we export setup, contact_start, move/swap, place/release, cleanup/reset. The good segment begins at contact_start and includes the object swap/release phase.", STYLES["Body"]),
        table(
            [
                ["Phase", "Meaning", "Signals"],
                ["setup", "synced but not useful for GT evaluation", "ego camera misses table/hands"],
                ["contact_start", "hands and objects enter the usable ego view", "hand masks near object masks"],
                ["move/swap", "ghee/salt are actively moved", "object centroid changes while hand contact holds"],
                ["place/release", "object settles after movement", "contact distance increases or motion slows"],
                ["cleanup/reset", "book/object adjustment or end clap", "reset marker / low manipulation"],
            ],
            [1.15 * inch, 2.4 * inch, 2.6 * inch],
        ),
        p("Current Status", STYLES["H1"]),
        table(
            [
                ["Component", "Status", "Notes"],
                ["Audio sync refinement", "working", "three markers agree within 0.02s"],
                ["Action-phase labels", "working draft", "exported for latest synced take"],
                ["GroundingDINO+SAM2 masks", "working", "hands/table/book objects on clean frames"],
                ["Prompted image generation", "working", "full-frame exo input, structured prompt"],
                ["Mask-level scoring", "working", "roles/contact/order/layout"],
                ["Arm-continuity scoring", "next patch", "needed to reject floating hands automatically"],
                ["Sparse geometry baseline", "weak", "single-view depth has too many holes"],
                ["HaMeR", "blocked", "needs MANO model and dedicated environment"],
            ],
            [1.55 * inch, 1.0 * inch, 3.8 * inch],
        ),
        p("Conclusion", STYLES["H1"]),
        p("The project is strongest when framed as a practical data engine, not as a magic reconstruction model. Extract physical state, generate candidates, and reject the ones that break manipulation constraints. The newest capture finally gives a clean sync and a small good evaluation window. The next polished demo should show the full generation sequence, then score only the good tail against paired ego.", STYLES["Body"]),
    ]

    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
