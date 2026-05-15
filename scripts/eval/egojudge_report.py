#!/usr/bin/env python3
"""Build an interpretable EgoJudge report with visual evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--scene-graph", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--cv-scores", required=True)
    parser.add_argument("--depth-scores", required=True)
    parser.add_argument("--vlm-scores", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rel(path: str | Path, base: Path) -> str:
    return str(Path(path).resolve().relative_to(base.resolve()))


def draw_source_evidence(source: Path, scene_graph: dict, out: Path) -> None:
    img = load_rgb(source)
    h, w = img.shape[:2]
    canvas = img.copy()
    colors = {
        "orange_carton_candidate": (255, 120, 30),
        "dark_mug_or_object_candidate": (70, 190, 255),
        "skin_hand_region_candidate": (120, 255, 120),
    }
    for name, ent in scene_graph.get("detected_entities", {}).items():
        if not isinstance(ent, dict) or "bbox_px" not in ent:
            continue
        x, y, bw, bh = [int(v) for v in ent["bbox_px"]]
        color = colors.get(name, (255, 255, 255))
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 3)
        cv2.putText(canvas, name.replace("_candidate", ""), (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for hand in scene_graph.get("detected_entities", {}).get("hands_from_phase2", []):
        x, y = [int(v) for v in hand.get("wrist_px", [0, 0])]
        cv2.circle(canvas, (x, y), 12, (255, 255, 0), -1)
        cv2.putText(canvas, f"{hand.get('handedness', 'hand')} wrist", (x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title("Perception evidence from source image")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_vlm(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    return {item["name"]: item for item in data.get("candidates", [])}


def build_grid(rows: list[dict], out: Path, max_items: int = 8) -> None:
    shown = rows[:max_items]
    cols = 2
    rows_n = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(14, 6 * rows_n))
    axes = np.array(axes).reshape(-1)
    for ax, row in zip(axes, shown):
        img = load_rgb(row["path"])
        ax.imshow(img)
        title = (
            f"{row['name']}\n"
            f"judge={row.get('mean_judge_score', '')} "
            f"pseudo={row.get('pseudo_reference_score', '')} "
            f"artifact={row.get('near_black_fraction', '')}"
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(shown) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def score_line(name: str, cv_scores: dict, depth_scores: dict, vlm_scores: dict) -> list[str]:
    lines = []
    cv_item = cv_scores.get("candidates", {}).get(name, {})
    depth_item = depth_scores.get("candidates", {}).get(name, {})
    vlm_item = vlm_scores.get(name, {})
    if cv_item:
        reasons = cv_item.get("evidence") or cv_item.get("reasons") or []
        lines.append(f"- CV/layout: **{cv_item.get('score')}**. Evidence: {'; '.join(reasons[:6])}.")
    if depth_item:
        reasons = depth_item.get("evidence") or depth_item.get("reasons") or []
        lines.append(f"- Depth/physics: **{depth_item.get('score')}**. Evidence: {'; '.join(reasons[:6])}.")
    if vlm_item:
        score = vlm_item.get("score")
        if isinstance(score, dict):
            vals = [v for v in score.values() if isinstance(v, (int, float))]
            score_text = round(float(np.mean(vals)), 2) if vals else score
        else:
            score_text = score
        strengths = "; ".join(vlm_item.get("strengths", [])[:3])
        failures = "; ".join(vlm_item.get("failures", [])[:3])
        lines.append(f"- VLM: **{score_text}**. Strengths: {strengths}. Failures: {failures}.")
    return lines


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path.cwd()

    scene_graph = json.loads(Path(args.scene_graph).read_text())
    metrics = json.loads(Path(args.metrics).read_text())
    cv_scores = json.loads(Path(args.cv_scores).read_text())
    depth_scores = json.loads(Path(args.depth_scores).read_text())
    vlm_scores = load_vlm(args.vlm_scores)

    evidence_path = out / "source_perception_evidence.png"
    grid_path = out / "candidate_grid.png"
    draw_source_evidence(Path(args.source), scene_graph, evidence_path)

    candidates = sorted(
        metrics.get("candidates", []),
        key=lambda r: r.get("combined_pseudo_gt_score", r.get("mean_judge_score", 0)) or 0,
        reverse=True,
    )
    build_grid(candidates, grid_path)

    lines = [
        "# EgoJudge Interpretable Verification Report",
        "",
        "This report explains what each verification step checks and shows the visual evidence used by the current implementation.",
        "",
        "## Source Perception",
        "",
        f"![source evidence]({rel(evidence_path, repo)})",
        "",
        "The source perception step extracts coarse but auditable cues: orange manipulated object, dark mug/object, skin/hand region, MediaPipe wrist points, and relative monocular depth summary.",
        "",
        "## Scorer Steps",
        "",
        "1. **CV/layout scorer** checks whether the candidate is a single image, has enough visual coverage, preserves the orange carton, preserves the dark mug, detects hands/skin, keeps carton left of mug, and keeps hands near the object.",
        "2. **Depth/physics scorer** runs monocular depth on each generated image, then checks hand-object depth consistency, object ordering, lower-frame hand placement, coverage, and sharpness.",
        "3. **VLM scorer** compares source and candidate images directly for ego-view plausibility, object identity, layout, hand-object interaction, and photorealism.",
        "4. **Reference metrics** compare against a real paired POV frame when available. In this demo they compare against a pseudo-reference ChatGPT image, so they are useful for debugging but not final scientific ground truth.",
        "5. **Artifact stats** catch cases where geometry baselines fool object detectors but still look bad: near-black fraction, entropy, and Laplacian sharpness.",
        "",
        "## Candidate Grid",
        "",
        f"![candidate grid]({rel(grid_path, repo)})",
        "",
        "## Ranked Candidates",
        "",
        "|rank|candidate|combined|judge|pseudo-ref|black frac|SSIM|edge F1|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for idx, row in enumerate(candidates, start=1):
        lines.append(
            "|"
            + "|".join(
                [
                    str(idx),
                    row["name"],
                    str(row.get("combined_pseudo_gt_score", "")),
                    str(row.get("mean_judge_score", "")),
                    str(row.get("pseudo_reference_score", "")),
                    str(row.get("near_black_fraction", "")),
                    str(row.get("ssim_vs_reference", "")),
                    str(row.get("edge_f1_vs_reference", "")),
                ]
            )
            + "|"
        )

    lines.extend(["", "## Per-Candidate Evidence", ""])
    for row in candidates:
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.extend(score_line(row["name"], cv_scores, depth_scores, vlm_scores) or ["- No scorer evidence found."])
        lines.append("")

    lines.extend(
        [
            "## Current Interpretation",
            "",
            "- The v2 geometry prompt is currently the best generative candidate because it explicitly constrains the bottom-of-frame red mat/body-side workspace and keeps the mug right of the carton.",
            "- The deterministic geometry baseline is valuable as a control, but it is not a physically correct reconstruction without true head/ego-camera pose.",
            "- The next real quantitative step is paired H2O data: exo input, ground-truth ego frame, and known camera metadata.",
            "",
        ]
    )

    (out / "interpretable_report.md").write_text("\n".join(lines))
    print(f"Wrote {out / 'interpretable_report.md'}")
    print(f"Wrote {evidence_path}")
    print(f"Wrote {grid_path}")


if __name__ == "__main__":
    main()
