#!/usr/bin/env python3
"""Quantitative EgoJudge metrics.

Supports two modes:
1. reference-based metrics if we have a real/pseudo ground-truth ego image
   (PSNR, SSIM, edge similarity, color histogram distance).
2. perception-based metrics that do not require GT
   (object/layout/contact/depth scores from existing EgoJudge reports).

These metrics are not meant to replace human inspection. They provide a
repeatable table for the report and make failure modes explicit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import wasserstein_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=None, help="Optional ego reference / pseudo-GT image.")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--cv-scores", default=None, help="Optional scores.json from egojudge_score.py")
    parser.add_argument("--depth-scores", default=None, help="Optional depth_scores.json from egojudge_depth_score.py")
    parser.add_argument("--vlm-scores", default=None, help="Optional vlm_scores.json from egojudge_vlm_score.py")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_to(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def psnr(ref: np.ndarray, cand: np.ndarray) -> float:
    ref_f = ref.astype(np.float32)
    cand_f = cand.astype(np.float32)
    mse = float(np.mean((ref_f - cand_f) ** 2))
    if mse < 1e-10:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_gray(ref: np.ndarray, cand: np.ndarray) -> float:
    """Simple global SSIM over grayscale images."""
    x = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    win = 11
    ux = uniform_filter(x, win)
    uy = uniform_filter(y, win)
    ux2 = ux * ux
    uy2 = uy * uy
    uxuy = ux * uy
    vx = uniform_filter(x * x, win) - ux2
    vy = uniform_filter(y * y, win) - uy2
    vxy = uniform_filter(x * y, win) - uxuy
    ssim_map = ((2 * uxuy + c1) * (2 * vxy + c2)) / ((ux2 + uy2 + c1) * (vx + vy + c2) + 1e-12)
    return float(np.mean(ssim_map))


def edge_f1(ref: np.ndarray, cand: np.ndarray) -> float:
    ref_e = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), 80, 160) > 0
    cand_e = cv2.Canny(cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY), 80, 160) > 0
    inter = float(np.logical_and(ref_e, cand_e).sum())
    prec = inter / max(float(cand_e.sum()), 1.0)
    rec = inter / max(float(ref_e.sum()), 1.0)
    return float(2 * prec * rec / max(prec + rec, 1e-8))


def hist_wasserstein(ref: np.ndarray, cand: np.ndarray) -> float:
    distances = []
    bins = np.arange(256)
    for ch in range(3):
        hr = cv2.calcHist([ref], [ch], None, [256], [0, 256]).ravel()
        hc = cv2.calcHist([cand], [ch], None, [256], [0, 256]).ravel()
        hr = hr / max(hr.sum(), 1)
        hc = hc / max(hc.sum(), 1)
        distances.append(wasserstein_distance(bins, bins, hr, hc))
    return float(np.mean(distances))


def near_black_fraction(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray < 12))


def image_entropy(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def laplacian_variance(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pseudo_reference_score(row: dict) -> float | None:
    if "ssim_vs_reference" not in row:
        return None
    ssim = float(row["ssim_vs_reference"])
    edge = float(row["edge_f1_vs_reference"])
    hist = float(row["rgb_hist_wasserstein"])
    hist_score = max(0.0, 1.0 - hist / 80.0)
    return round(100.0 * (0.50 * ssim + 0.30 * edge + 0.20 * hist_score), 2)


def load_score_map(path: str | None, kind: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    if kind == "vlm":
        scores = {}
        for item in data.get("candidates", []):
            score = item.get("score")
            if isinstance(score, dict):
                vals = [v for v in score.values() if isinstance(v, (int, float))]
                scores[item["name"]] = round(float(np.mean(vals)), 2) if vals else None
            else:
                scores[item["name"]] = score
        return scores
    return {name: item.get("score") for name, item in data.get("candidates", {}).items()}


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reference = load_rgb(args.reference) if args.reference else None
    cv_scores = load_score_map(args.cv_scores, "cv")
    depth_scores = load_score_map(args.depth_scores, "depth")
    vlm_scores = load_score_map(args.vlm_scores, "vlm")

    rows = []
    for candidate in args.candidates:
        path = Path(candidate)
        if not path.exists():
            continue
        img = load_rgb(str(path))
        row = {"name": path.name, "path": str(path.resolve())}
        if reference is not None:
            cand = resize_to(img, reference.shape[:2])
            row.update(
                {
                    "psnr_vs_reference": round(psnr(reference, cand), 3),
                    "ssim_vs_reference": round(ssim_gray(reference, cand), 4),
                    "edge_f1_vs_reference": round(edge_f1(reference, cand), 4),
                    "rgb_hist_wasserstein": round(hist_wasserstein(reference, cand), 3),
                }
            )
            row["pseudo_reference_score"] = pseudo_reference_score(row)
        row["near_black_fraction"] = round(near_black_fraction(img), 4)
        row["entropy"] = round(image_entropy(img), 3)
        row["laplacian_variance"] = round(laplacian_variance(img), 2)
        row["cv_layout_score"] = cv_scores.get(path.name)
        row["depth_physics_score"] = depth_scores.get(path.name)
        row["vlm_score"] = vlm_scores.get(path.name)

        numeric = [
            row.get("cv_layout_score"),
            row.get("depth_physics_score"),
            row.get("vlm_score"),
        ]
        numeric = [v for v in numeric if isinstance(v, (int, float))]
        if numeric:
            row["mean_judge_score"] = round(float(np.mean(numeric)), 2)
            artifact_penalty = min(35.0, 100.0 * row["near_black_fraction"])
            row["artifact_adjusted_score"] = round(max(0.0, row["mean_judge_score"] - artifact_penalty), 2)
            if row.get("pseudo_reference_score") is not None:
                row["combined_pseudo_gt_score"] = round(
                    max(0.0, 0.60 * row["mean_judge_score"] + 0.40 * row["pseudo_reference_score"] - artifact_penalty),
                    2,
                )
        rows.append(row)

    result = {"reference": str(Path(args.reference).resolve()) if args.reference else None, "candidates": rows}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    headers = [
        "name",
        "psnr_vs_reference",
        "ssim_vs_reference",
        "edge_f1_vs_reference",
        "rgb_hist_wasserstein",
        "pseudo_reference_score",
        "near_black_fraction",
        "entropy",
        "laplacian_variance",
        "cv_layout_score",
        "depth_physics_score",
        "vlm_score",
        "mean_judge_score",
        "artifact_adjusted_score",
        "combined_pseudo_gt_score",
    ]
    lines = ["# EgoJudge Quantitative Metrics", "", "|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join("" if row.get(h) is None else str(row.get(h, "")) for h in headers) + "|")
    (out / "metrics.md").write_text("\n".join(lines))
    print(f"Wrote {out / 'metrics.json'}")
    print(f"Wrote {out / 'metrics.md'}")


if __name__ == "__main__":
    main()
