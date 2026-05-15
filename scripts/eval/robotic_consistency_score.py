#!/usr/bin/env python3
"""Build an interpretable robotics scorecard for EgoJudge outputs.

This script intentionally does not pretend to certify physical truth. It
aggregates the diagnostics we can actually measure right now: paired image
metrics, hand-landmark observability, monocular-depth sanity, and paired
view-overlap calibration. The output is meant for the report and for choosing
which candidates are safe enough to show/use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-depth-report", required=True)
    parser.add_argument("--paired-metrics", default=None)
    parser.add_argument("--view-calibration-report", default=None)
    parser.add_argument("--scene-graph", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def load_json(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def hand_entry_score(item: dict) -> tuple[float, str]:
    hands = item.get("hands", [])
    if not hands:
        return 0.0, "no hands detected"

    image_h = None
    if item.get("image_shape_hw"):
        image_h = float(item["image_shape_hw"][0])
    wrist_y = []
    widths = []
    heights = []
    for hand in hands:
        landmarks = hand.get("landmarks_px") or []
        if not landmarks:
            continue
        if image_h:
            wrist_y.append(hand["wrist_px"][1] / image_h)
        else:
            max_y = max(pt[1] for pt in landmarks)
            if max_y > 0:
                wrist_y.append(hand["wrist_px"][1] / max_y)
        box = hand.get("bbox_px") or [0, 0, 0, 0]
        widths.append(box[2])
        heights.append(box[3])

    if not wrist_y:
        return 0.25, "hands detected but wrist landmarks missing"
    median_y = sorted(wrist_y)[len(wrist_y) // 2]
    lower_half = clamp01((median_y - 0.35) / 0.35)
    size_ok = 1.0 if widths and heights and min(widths) > 40 and min(heights) > 40 else 0.5
    return round(0.75 * lower_half + 0.25 * size_ok, 3), f"median wrist y proxy={median_y:.2f}; hands enter from lower/side frame if score is high"


def hand_count_score(item: dict, expected_min: int, expected_max: int) -> tuple[float, str]:
    count = int(item.get("hands_detected", 0))
    if expected_min <= count <= expected_max:
        return 1.0, f"{count} hands detected, inside expected range {expected_min}-{expected_max}"
    distance = min(abs(count - expected_min), abs(count - expected_max))
    score = clamp01(1.0 - distance / max(expected_max, 1))
    return round(score, 3), f"{count} hands detected, outside expected range {expected_min}-{expected_max}"


def depth_sanity_score(item: dict, gt_depth: dict | None) -> tuple[float, str]:
    d = item.get("depth_summary") or {}
    if not d:
        return 0.0, "no depth map available"
    span = float(d.get("max", 0.0)) - float(d.get("min", 0.0))
    span_score = clamp01(span / 5.0)
    if gt_depth:
        gt_span = float(gt_depth.get("max", 0.0)) - float(gt_depth.get("min", 0.0))
        gt_med = float(gt_depth.get("median", 0.0))
        med = float(d.get("median", 0.0))
        span_match = clamp01(1.0 - abs(span - gt_span) / max(gt_span, 1e-6))
        med_match = clamp01(1.0 - abs(med - gt_med) / max(abs(gt_med), 1e-6))
        score = 0.35 * span_score + 0.35 * span_match + 0.30 * med_match
        reason = f"depth span={span:.2f}; GT span={gt_span:.2f}; median={med:.2f}, GT median={gt_med:.2f}"
    else:
        score = span_score
        reason = f"depth span={span:.2f}; no GT depth comparison"
    return round(score, 3), reason


def paired_score(metric_row: dict) -> tuple[float | None, str]:
    if not metric_row:
        return None, "no paired GT metric for this candidate"
    ssim = float(metric_row.get("ssim", 0.0))
    edge = float(metric_row.get("edge_f1", 0.0))
    hist = float(metric_row.get("rgb_hist_wasserstein", 80.0))
    psnr = float(metric_row.get("psnr", 0.0))
    hist_score = clamp01(1.0 - hist / 80.0)
    score = 100.0 * (0.45 * ssim + 0.20 * edge + 0.25 * hist_score + 0.10 * min(psnr / 30.0, 1.0))
    return round(score, 2), f"paired score from SSIM={ssim:.3f}, edge F1={edge:.4f}, RGB hist W={hist:.2f}, PSNR={psnr:.2f}"


def camera_overlap_score(report: dict) -> tuple[float | None, str]:
    if not report:
        return None, "no paired view-overlap report"
    inliers = int(report.get("inliers", 0))
    ratio = float(report.get("inlier_ratio", 0.0))
    score = 100.0 * (0.7 * clamp01(ratio) + 0.3 * clamp01(inliers / 50.0))
    return round(score, 2), f"{inliers} RANSAC inliers, inlier ratio={ratio:.3f}; interpretation={report.get('interpretation', 'unknown')}"


def candidate_key(path: str) -> str:
    return Path(path).name


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hand_depth = load_json(args.hand_depth_report)
    paired = load_json(args.paired_metrics).get("results", [])
    view_report = load_json(args.view_calibration_report)
    scene = load_json(args.scene_graph)

    images = hand_depth.get("images", [])
    by_name = {item["name"]: item for item in images}
    gt_depth = (by_name.get("ego_gt") or {}).get("depth_summary")
    exo_count = int((by_name.get("exo") or {}).get("hands_detected", 0))
    scene_hands = len(scene.get("hands", [])) if scene else 0

    # MediaPipe often misses occluded/contact hands in exo frames. Use it as
    # the lower-confidence landmark signal, but allow the VLM scene graph to
    # widen the expected hand range when it explicitly sees more hands.
    if exo_count > 0 or scene_hands > 0:
        expected_min = max(1, min(exo_count or scene_hands, scene_hands or exo_count, 2))
        expected_max = max(exo_count, scene_hands, expected_min)
        expected_max = min(expected_max, 2 if max(exo_count, scene_hands) <= 2 else 3)
    else:
        expected_min = 0
        expected_max = 0
    paired_by_candidate = {candidate_key(row.get("candidate", row.get("candidate_name", ""))): row for row in paired}

    camera_score, camera_reason = camera_overlap_score(view_report)
    cards = []
    for item in images:
        name = item["name"]
        if name in {"exo", "ego_gt"}:
            continue
        path_name = candidate_key(item["path"])
        metric_row = paired_by_candidate.get(path_name, {})
        p_score, p_reason = paired_score(metric_row)
        hc_score, hc_reason = hand_count_score(item, expected_min, expected_max)
        he_score, he_reason = hand_entry_score(item)
        d_score, d_reason = depth_sanity_score(item, gt_depth)

        robotics_score = 100.0 * (0.35 * hc_score + 0.20 * he_score + 0.25 * d_score)
        if camera_score is not None:
            robotics_score += 0.20 * camera_score
        if p_score is not None:
            final_score = 0.55 * p_score + 0.45 * robotics_score
        else:
            final_score = robotics_score

        hand_count = int(item.get("hands_detected", 0))
        if hand_count > expected_max:
            verdict = "reject or downweight: likely extra-hand hallucination"
        elif hand_count < expected_min:
            verdict = "reject or downweight: missing visible hand/action evidence"
        elif p_score is not None and p_score < 40:
            verdict = "downweight: plausible but not aligned to paired ego GT"
        else:
            verdict = "candidate usable for qualitative demo; still not ground truth"

        cards.append(
            {
                "name": name,
                "path": item["path"],
                "candidate_file": path_name,
                "paired_score": p_score,
                "paired_reason": p_reason,
                "hand_count_score": round(100.0 * hc_score, 1),
                "hand_count_reason": hc_reason,
                "hand_entry_score": round(100.0 * he_score, 1),
                "hand_entry_reason": he_reason,
                "depth_sanity_score": round(100.0 * d_score, 1),
                "depth_sanity_reason": d_reason,
                "camera_overlap_score": camera_score,
                "camera_overlap_reason": camera_reason,
                "robotics_score": round(robotics_score, 2),
                "final_score": round(final_score, 2),
                "verdict": verdict,
            }
        )

    result = {
        "expected_hand_range": [expected_min, expected_max],
        "source_exo_hands_detected": exo_count,
        "scene_graph_hands": scene_hands,
        "ground_truth_hand_detector_note": "GT hand count is used only as observability context; low-light GT can make detector fail.",
        "camera_overlap": {"score": camera_score, "reason": camera_reason},
        "candidates": cards,
    }
    (out / "robotic_consistency_score.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# Robotic Consistency Scorecard",
        "",
        f"Expected hand range: {expected_min}-{expected_max}. Exo detector saw {exo_count}; VLM scene graph listed {scene_hands}.",
        "",
        "|candidate|final|paired|hands|entry|depth|camera|verdict|",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for card in cards:
        paired_text = "" if card["paired_score"] is None else str(card["paired_score"])
        camera_text = "" if card["camera_overlap_score"] is None else str(card["camera_overlap_score"])
        lines.append(
            f"|{card['name']}|{card['final_score']}|{paired_text}|{card['hand_count_score']}|{card['hand_entry_score']}|{card['depth_sanity_score']}|{camera_text}|{card['verdict']}|"
        )
    lines += ["", "## Evidence"]
    for card in cards:
        lines += [
            "",
            f"### {card['name']}",
            f"- Paired: {card['paired_reason']}",
            f"- Hands: {card['hand_count_reason']}; {card['hand_entry_reason']}",
            f"- Depth: {card['depth_sanity_reason']}",
            f"- Camera: {card['camera_overlap_reason']}",
        ]
    (out / "robotic_consistency_score.md").write_text("\n".join(lines))
    print(f"Wrote {out / 'robotic_consistency_score.json'}")
    print(f"Wrote {out / 'robotic_consistency_score.md'}")


if __name__ == "__main__":
    main()
