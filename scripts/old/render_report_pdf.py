#!/usr/bin/env python3
"""Render a detailed project PDF without requiring a LaTeX installation.

The machine currently has no pdflatex/tectonic/xelatex, so this renderer uses
ReportLab while keeping paper/report.tex as a LaTeX source draft.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "report.pdf"
PROMPT_PATH = ROOT / "experiments/mendeley_two_person_dim_dice/vlm_scene_00000001/prompt_conditioned_vlm.txt"
SCENE_PATH = ROOT / "experiments/mendeley_two_person_dim_dice/vlm_scene_00000001/vlm_scene_graph.json"


def safe(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def para(text: str, style):
    return Paragraph(safe(text), style)


def bullet(items: list[str], style):
    return [Paragraph("&bull; " + safe(item), style) for item in items]


def fig(path: str, width: float = 6.6 * inch, ratio: float = 0.58):
    p = ROOT / path
    if not p.exists():
        return para(f"[missing figure: {path}]", getSampleStyleSheet()["BodyText"])
    return Image(str(p), width=width, height=width * ratio)


def table(rows, col_widths=None):
    t = Table(rows, hAlign="LEFT", colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8f8")]),
            ]
        )
    )
    return t


def code_block(text: str, style, max_chars: int | None = None):
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated in PDF; full prompt is in the repository file]"
    return Preformatted(text, style)


def main() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, leading=17, spaceBefore=10, spaceAfter=5)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11.5, leading=14, spaceBefore=7, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.6, leading=12.5, spaceAfter=6)
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8.2, leading=10.5, textColor=colors.darkslategray)
    mono = ParagraphStyle("Mono", parent=styles["Code"], fontName="Courier", fontSize=6.6, leading=8.0)

    prompt_text = PROMPT_PATH.read_text() if PROMPT_PATH.exists() else "[prompt file missing]"
    scene = json.loads(SCENE_PATH.read_text()) if SCENE_PATH.exists() else {}

    story = []
    story += [
        para("EgoJudge: Perception-Conditioned Selection for Generative Exo-to-Ego View Synthesis", title),
        para("Robotics Final Project Report Draft", styles["Heading3"]),
        para(
            "This project asks whether we can turn third-person manipulation imagery into useful first-person observations for robot learning. "
            "A pure reconstruction pipeline is underconstrained because the true head/camera pose and occluded surfaces are not known. "
            "A pure image-generation pipeline looks good but may hallucinate objects, contacts, and geometry. EgoJudge is the hybrid: extract a scene graph, generate multiple ego-view candidates, and select/evaluate them with geometry-aware and robot-data-aware checks.",
            body,
        ),
    ]

    story += [
        para("1. Motivation: Why This Matters For Robot Learning", h1),
        para(
            "Modern Vision-Language-Action systems such as Physical Intelligence-style robot foundation models rely on large, heterogeneous visual-action datasets. "
            "The data bottleneck is not only language labels; it is varied embodied visual experience. Human egocentric videos are attractive because they show hands, objects, contacts, and task phases from a policy-relevant viewpoint. "
            "However, egocentric data is harder to collect than ordinary third-person video. If exocentric videos could be converted into plausible ego observations, they could become additional pretraining, augmentation, or supervision data for VLA/RL pipelines.",
            body,
        ),
        para(
            "For a humanoid robot policy, the generated image does not need to be pixel-perfect art. It needs to preserve the facts that a policy uses: object identity and affordance, reachable layout, hand-object contact, action phase, and temporal stability. A beautiful frame that moves the cup to the wrong side or invents extra hands is bad training data.",
            body,
        ),
    ]

    story += [
        para("2. Problem Formulation", h1),
        para("Input: an exocentric image or frame I_exo of a manipulation scene. Output: one or more candidate egocentric images I_ego^k plus confidence/evaluation scores.", body),
        para(
            "The core difficulty is that exo-to-ego is not a standard image translation problem. It requires changing the camera center and pitch while preserving contact, object identity, and scene layout. The target view may contain surfaces not visible in the source, so some hallucination is unavoidable. EgoJudge therefore does not pretend every output is physically verified; it explicitly scores and filters outputs.",
            body,
        ),
    ]

    story += [
        para("3. What We Tried First: Sparse Geometry", h1),
        para(
            "The first implementation followed a geometry-first intuition: estimate monocular depth, backproject into a point cloud, choose a virtual eye camera, and reproject into an ego view. "
            "This failed on the initial tabletop image. The red mat/table region appeared in the wrong part of the frame and the view was warped because the virtual camera point was guessed. "
            "The old Phase 3 script even created fake ego hand correspondences, making the transform arbitrary. We now keep this as a failure baseline.",
            body,
        ),
        fig("experiments/egojudge/img7795/standard_baseline/standard_baseline_debug.png", ratio=0.28),
        para("Figure 1. Geometry-only reprojection baseline. It preserves some coarse structure but is not a valid ego camera image.", small),
    ]

    story += [
        para("4. EgoJudge Method", h1),
        para("EgoJudge is a generate-and-select pipeline:", body),
        *bullet(
            [
                "Perception extraction: detect workspace, hands, objects, contacts, and camera hints.",
                "Prompt construction: convert these measurements into explicit generation constraints.",
                "Candidate generation: produce K variants from an image model.",
                "Candidate judging: evaluate paired similarity, object/layout/contact preservation, physical plausibility, and artifact risk.",
                "Selection: choose the highest-confidence ego candidate and report why it was selected.",
            ],
            body,
        ),
        para("The key design choice is interpretability: each preprocessing output is visible as a JSON scene graph and an overlay, and each score is tied to a concrete failure mode.", body),
    ]

    story += [
        para("5. Preprocessing: How Each Cue Improves The Prompt", h1),
        para(
            "The first HSV-based scripts were useful for debugging but too hardcoded. The current general path is scripts/vlm_scene_graph.py: it asks a VLM for a structured scene graph with normalized boxes and relations. "
            "We then optionally refine the VLM boxes into approximate masks using GrabCut in scripts/refine_vlm_scene_masks.py. In a stronger future version, these masks should come from GroundingDINO or OWLv2 plus SAM2.",
            body,
        ),
        table(
            [
                ["Preprocessing cue", "Used in prompt as", "Why it matters"],
                ["Workspace/table", "Target surface, near/far orientation, body-side bottom", "Keeps the ego view from becoming a random side view; defines reachable area."],
                ["Hands/forearms", "Where hands enter from, contact state, grasp/resting state", "Robot policies need contact/action phase; extra or missing hands corrupt supervision."],
                ["Manipulated object", "Object identity, role, contact_with_hand", "Preserves affordance and prevents the generator from changing the task."],
                ["Relative layout", "left/right/near/far constraints", "Keeps training labels consistent with action geometry."],
                ["Failure risks", "Explicit negative instructions", "Prevents common hallucinations: moved object, extra hands, wrong camera angle."],
                ["Camera hints", "body_side_in_target, far_side_in_target, likely ego view", "Directly targets the hardest part: the POV and downward angle."],
            ],
            [1.35 * inch, 2.1 * inch, 3.0 * inch],
        ),
        Spacer(1, 0.08 * inch),
        fig("experiments/mendeley_two_person_dim_dice/vlm_scene_00000001/vlm_scene_graph_overlay.png", width=4.4 * inch, ratio=1.48),
        para("Figure 2. VLM scene graph overlay. Boxes are approximate, but unlike HSV rules they are semantic and object-level.", small),
        fig("experiments/mendeley_two_person_dim_dice/vlm_scene_00000001/refined_masks/refined_mask_overlay.png", width=4.4 * inch, ratio=1.48),
        para("Figure 3. VLM boxes refined with GrabCut. This more precisely highlights the workspace, cup, and hand regions, but should eventually be replaced with SAM2 masks.", small),
    ]

    story += [
        PageBreak(),
        para("6. Exact Final Prompt", h1),
        para("The final VLM-conditioned prompt used for the next generation round is below. The full file is experiments/mendeley_two_person_dim_dice/vlm_scene_00000001/prompt_conditioned_vlm.txt.", body),
        code_block(prompt_text, mono, max_chars=4600),
    ]

    story += [
        para("7. How The Evaluation Actually Works", h1),
        para("There are three categories of evaluation: paired GT metrics, scene/robotics checks, and temporal checks.", body),
        para("7.1 Paired GT Metrics", h2),
        para(
            "When a real ego frame exists, paired_eval_report.py matches a generated candidate to the same frame id, loads the GT ego image, resizes the generated image to the GT dimensions, and computes PSNR, SSIM, Canny edge F1, and RGB histogram Wasserstein distance. "
            "The visual report letterboxes all images to a common canvas for readability, but the metrics use resized candidate versus GT.",
            body,
        ),
        table(
            [
                ["Metric", "What it checks", "Why it matters", "Failure mode"],
                ["PSNR", "Pixel-level intensity error", "Catches gross color/lighting mismatch", "Too harsh for different but plausible POV crops."],
                ["SSIM", "Local structural similarity", "Rewards table/object structure and contrast alignment", "Still penalizes valid viewpoint differences."],
                ["Edge F1", "Overlap of Canny edges", "Detects whether object/table boundaries align", "Very low if crop/scale differs."],
                ["RGB hist W", "Color distribution distance", "Catches lighting/background/style drift", "Cannot know whether the cup moved."],
                ["VLM judge", "Semantic layout/contact/photo plausibility", "Closer to robot-data usefulness", "Subjective/model-dependent."],
                ["Artifact stats", "black holes, entropy, sharpness", "Filters sparse/debug/broken outputs", "Does not prove semantic correctness."],
            ],
            [0.9 * inch, 1.45 * inch, 2.1 * inch, 2.0 * inch],
        ),
        Spacer(1, 0.08 * inch),
        fig("experiments/mendeley_two_person_dim_dice/eval/paired_visual_report.png", ratio=0.36),
        para("Figure 4. Paired evaluation against real ego GT. The generated candidate is cleaner and plausible, but not exact-pose aligned.", small),
        table(
            [
                ["Frame", "Candidate", "PSNR", "SSIM", "Edge F1", "RGB Hist W"],
                ["00000001", "OpenAI safe prompt", "14.87", "0.534", "0.0026", "8.20"],
                ["00000001", "OpenAI VLM scene graph", "12.33", "0.476", "0.0035", "29.02"],
            ],
            [0.8 * inch, 1.7 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch, 0.9 * inch],
        ),
        para(
            "The VLM-scene prompt generated a sharper, cleaner image but scored worse against the actual paired ego frame because it selected a different crop/FOV and moved away from the true low-light phone-camera view. This is not just a failure; it shows why EgoJudge is needed. Good-looking samples need to be selected against measurable camera/layout consistency, not only visual appeal.",
            body,
        ),
        para("7.2 Hand and Depth Diagnostics", h2),
        para(
            "The evaluator now includes an explicit hand/depth diagnostic. hand_depth_diagnostic.py runs MediaPipe Tasks HandLandmarker on the exo, generated, and GT images and runs Depth Anything V2 Small for relative depth. "
            "On the Mendeley Dice example it detects 2 hands in the exo image, 2 hands in the safer generated image, 3 hands in the VLM-scene generated image, and 0 hands in the dark/blurry GT ego image. This exposes two real issues: image models can invent extra hands, and even ground truth can be hard for an off-the-shelf detector. The detector result should therefore be interpreted as an observability signal, not absolute truth.",
            body,
        ),
        fig("experiments/mendeley_two_person_dim_dice/hand_depth_compare_00000001/hand_depth_diagnostic.png", ratio=0.98),
        para(
            "Figure 5. Hand and depth diagnostics. The VLM-scene candidate is visually cleaner, but it is penalized because the hand detector sees an extra hand.",
            small,
        ),
        para("7.3 Camera/View-Overlap Diagnostic", h2),
        para(
            "view_calibration_diagnostic.py uses ORB feature matching and RANSAC homography between paired exo and ego frames. This estimates the planar overlap of the ego view inside the exo image. "
            "For the first Dice frame, it found 537 exo keypoints, 343 ego keypoints, 170 good matches, 166 inliers, and an inlier ratio of 0.976. This means the table-cover texture has enough overlap for a rough planar diagnostic. It is not a full 6-DoF camera pose, but it gives concrete evidence about the view relation and can be used for calibration when paired data is available.",
            body,
        ),
        fig("experiments/mendeley_two_person_dim_dice/view_calibration_00000001/view_overlap_diagnostic.png", ratio=0.36),
        para("Figure 6. Paired view-overlap diagnostic. Yellow polygon is the projected ego footprint in the exo frame under a planar homography.", small),
    ]

    story += [
        para("8. Generating 10 Variants And Selecting The Best", h1),
        para(
            "The selection workflow is intentionally conservative. generate_variants.py generates up to K=10 candidates and records a manifest. "
            "select_best_variant.py currently ranks candidates with a weighted paired score: 45 percent SSIM, 20 percent edge F1, 25 percent color histogram similarity, 10 percent normalized PSNR, minus artifact penalty. "
            "robotic_consistency_score.py then adds hand-count, hand-entry, depth-sanity, and camera-overlap evidence. This is the part that makes EgoJudge more than a pretty-image chooser.",
            body,
        ),
        para(
            "Ablations should test: generic prompt, VLM scene-graph prompt, scene graph without failure risks, scene graph without camera hints, and scene graph plus mask/overlay conditioning. The expected result is that camera hints improve POV, object constraints improve identity, and failure-risk text reduces extra hands or object drift.",
            body,
        ),
        table(
            [
                ["Ablation", "Prompt content", "Hypothesis"],
                ["Generic", "Only asks for first-person view", "Looks plausible but moves objects/changes layout."],
                ["Scene graph", "Objects, boxes, contact, layout", "Better object and relation preservation."],
                ["No camera hints", "Scene graph without body-side/far-side", "More wrong POV or top/bottom flips."],
                ["No failure risks", "No explicit negative constraints", "More extra hands/changed objects."],
                ["Mask/overlay conditioning", "Scene graph plus visual control image", "Best object localization if model respects image cues."],
            ],
            [1.2 * inch, 2.2 * inch, 3.0 * inch],
        ),
        para(
            "The current mini-ablation already shows a warning: the VLM scene-graph prompt made a cleaner image but did not beat the simpler safe prompt under paired GT metrics. This suggests the next prompt should preserve the dataset camera style more aggressively: low-light, chest-mounted/phone POV, narrower crop, and no beautification.",
            body,
        ),
        table(
            [
                ["Candidate", "Final", "Paired", "Hands", "Entry", "Depth", "Verdict"],
                ["Safe prompt", "69.24", "51.47", "100.0", "77.7", "83.0", "usable demo; not ground truth"],
                ["VLM scene graph", "55.49", "41.51", "50.0", "71.0", "84.8", "reject/downweight: extra-hand risk"],
            ],
            [1.25 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch, 2.4 * inch],
        ),
        para(
            "The scorecard is deliberately interpretable. The VLM scene-graph candidate loses because it has three detected hands while the exo hand-landmark detector saw two. That matters for robot learning: a frame with extra limbs teaches the wrong contact/action state even if it looks like a nicer photograph.",
            body,
        ),
    ]

    story += [
        para("8.1 New-Frame Generalization And Prompt Ablation", h1),
        para(
            "I extracted five additional paired Mendeley frames directly from the nested archive: 00000641, 00000721, 00000801, 00000881, and 00000961. "
            "The planar view-overlap diagnostic remained strong on all five frames, with RANSAC inlier ratios from 0.980 to 0.994. Hand detection was noisier: exo frames reported 1-3 hands, while dark ego GT frames sometimes reported zero.",
            body,
        ),
        fig("experiments/mendeley_two_person_dim_dice_more5/montage.png", width=4.2 * inch, ratio=1.65),
        para("Figure 7. Five additional paired exo/ego frames pulled from the nested Mendeley archive.", small),
        table(
            [
                ["Frame", "Exo kp", "Ego kp", "Matches", "Inliers", "Ratio"],
                ["00000641", "721", "481", "101", "99", "0.980"],
                ["00000721", "803", "545", "106", "104", "0.981"],
                ["00000801", "583", "398", "179", "177", "0.989"],
                ["00000881", "532", "334", "162", "161", "0.994"],
                ["00000961", "566", "456", "185", "182", "0.984"],
            ],
            [0.9 * inch, 0.7 * inch, 0.7 * inch, 0.8 * inch, 0.8 * inch, 0.7 * inch],
        ),
        para(
            "On frame 00000641, a direct safe prompt with explicit hand/person wording was blocked by the image API safety system. Sanitized prompts succeeded, which is an important engineering finding: robust pipelines need failure logging and fallback prompt templates.",
            body,
        ),
        fig("experiments/mendeley_two_person_dim_dice_more5/eval_ablation_00000641/paired_visual_report.png", ratio=0.58),
        para("Figure 8. Prompt ablation outputs for frame 00000641 versus paired GT ego.", small),
        table(
            [
                ["Candidate", "PSNR", "SSIM", "Edge F1", "RGB W", "Robotic verdict"],
                ["api_sanitized", "14.96", "0.512", "0.0076", "14.19", "downweight: extra hand"],
                ["objects_camera_sanitized", "15.01", "0.505", "0.0041", "17.79", "best current compromise"],
                ["strong_pov_sanitized", "15.36", "0.539", "0.0079", "19.14", "downweight: missing hand evidence"],
            ],
            [1.55 * inch, 0.55 * inch, 0.55 * inch, 0.65 * inch, 0.55 * inch, 2.4 * inch],
        ),
        para(
            "Interpretation: stronger POV wording improves paired SSIM/PSNR, so camera wording matters. But it can lose manipulation evidence: the strong POV output only had one detected hand. The object+camera prompt is currently the best compromise because it preserves the expected two-hand structure.",
            body,
        ),
    ]

    story += [
        para("9. The Camera POV Problem", h1),
        para(
            "The largest remaining error is exact ego camera point and pitch. A head detector can help only when the head is visible. It can estimate a rough eye center from a head or face box, but it does not solve yaw/pitch/roll, body posture, or the fact that head-mounted/chest-mounted cameras are offset from the eyes. "
            "In our initial exo image the face/eyes were blacked out; in the Mendeley overhead data the ego camera is chest-mounted while the exo camera is tripod-mounted, so head detection is not enough.",
            body,
        ),
        para(
            "Known camera metadata helps but does not fully solve calibration. The Mendeley dataset reports synchronized ego/exo iPhone video, with one phone chest-mounted at about 60 degrees and another on a tripod at about 30 degrees. That gives a rough camera model, but not the exact extrinsics for a frame. "
            "H2O is better because it has calibrated RGB-D multi-view cameras and a head-mounted ego view, but the full subject download is too large for local iteration.",
            body,
        ),
        para(
            "Best practical auto-calibration for tomorrow: add visible fiducials. Four AprilTags or checkerboard corners on the table let us estimate table plane and body-side direction in both views. Modern 3D foundation models such as DUSt3R, MASt3R, or VGGT may estimate camera pose from overlapping views, but dynamic hands, low texture, and exo/ego viewpoint gap make fiducials more reliable for a two-day project.",
            body,
        ),
        table(
            [
                ["Approach", "What it gives", "Problem"],
                ["Head/face detector", "rough head/eye location in exo", "fails if face hidden; does not give camera pitch/FOV/chest offset"],
                ["Body pose / SMPL", "torso/head orientation prior", "heavy setup and still not exact camera extrinsics"],
                ["Feature homography", "paired planar overlap", "needs paired frames or shared texture; assumes planar scene"],
                ["AprilTags/checkerboard", "reliable table pose and scale", "requires collecting our own data with markers"],
                ["DUSt3R/MASt3R/VGGT", "possible camera/depth reconstruction", "less reliable under dynamic hands and large ego/exo viewpoint gap"],
            ],
            [1.35 * inch, 2.0 * inch, 3.1 * inch],
        ),
    ]

    story += [
        para("10. Why This Is Useful For VLA/RL Training", h1),
        para(
            "For robot data, EgoJudge can act as a filter and annotator rather than blindly adding generated frames. The output should include not only images but also confidence: object preserved, contact preserved, layout preserved, temporal stable. "
            "Low-confidence generated frames should be excluded or used only for qualitative demos. High-confidence frames can augment visual pretraining or train perception modules to recognize task state from ego view.",
            body,
        ),
        *bullet(
            [
                "Object identity and affordance: the policy must know what object is being manipulated.",
                "Contact/action state: reaching versus grasping versus releasing changes the action label.",
                "Reachable layout: spatial relations determine feasible robot motion.",
                "Temporal consistency: policy rollouts cannot learn from frames where objects jump.",
                "Confidence labels: generated data should be weighted or filtered, not treated as ground truth by default.",
            ],
            body,
        ),
    ]

    story += [
        para("10.1 Training / Fine-Tuning Path", h1),
        para(
            "The realistic two-day version is not full image-model fine-tuning. The feasible training-like contribution is a data engine: generate candidates, score them with EgoJudge, and save preference labels. "
            "Those tuples can train a lightweight ranker now and could later support LoRA fine-tuning of an open image-to-image diffusion model. Online RL on a large image/video model is too heavy for this project, but rejection sampling and DPO-style preference data are credible future work.",
            body,
        ),
        *bullet(
            [
                "Immediate: prompt optimization using EgoJudge scores.",
                "Near-term: preference dataset of source frame, prompt, candidates, and judge scores.",
                "Future: LoRA on an open diffusion image-to-image model for camera/crop style.",
                "Future: VLM/LLM judge as reward model for offline preference optimization, not online RL in this deadline.",
            ],
            body,
        ),
        para(
            "Implemented artifact: scripts/export_preference_dataset.py writes pairwise JSONL preferences. For frame 00000641 it produced two usable preferences choosing objects_camera_sanitized over the extra-hand and missing-hand alternatives.",
            body,
        ),
    ]

    story += [
        para("11. Next Work Plan", h1),
        para(
            "Dataset scan update: I scanned all local Mendeley exocentric splits. They are mostly overhead tabletop views with cropped arms/hands. Some 3/4-person white/yellow-light frames show partial blurred faces or upper bodies, but there is no clean one-person split with full body, visible head, and hands. Therefore Mendeley is useful for paired evaluator sanity checks, not for proving head/eye ego-camera recovery.",
            body,
        ),
        *bullet(
            [
                "Collect our own paired data tomorrow: exo ZV-E10/iPhone plus iPhone POV, clap sync, 20-50 selected frames.",
                "Place AprilTags/checkerboard/fiducials on the table so camera orientation can be estimated.",
                "Run VLM scene graph extraction on every selected exo frame.",
                "Generate 3 variants on two frames, inspect, then generate 10 variants on final frames.",
                "Run paired metrics, VLM judge, object/contact checks, and temporal consistency.",
                "Write final ablation table: generic prompt vs scene graph vs scene graph with camera hints vs scene graph with mask conditioning.",
                "Keep geometry-only sparse reprojection as a baseline/failure analysis, not the claimed solution.",
            ],
            body,
        ),
        para("12. References", h1),
        *bullet(
            [
                "Physical Intelligence pi0 blog and pi0.5 technical report: broad VLA generalization depends on heterogeneous robot, web, and semantic data mixtures.",
                "EgoWorld project page: single-image exo-to-ego using sparse RGB map, egocentric 3D hand pose, and text-conditioned diffusion.",
                "Mendeley egocentric/exocentric hand dataset: synchronized iPhone ego/exo paired frames across games and lighting conditions.",
                "H2O dataset: synchronized multi-view RGB-D, interaction labels, hand/object poses, camera poses, meshes, and scene point clouds.",
            ],
            body,
        ),
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
