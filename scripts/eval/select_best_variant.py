#!/usr/bin/env python3
"""Rank generated variants using paired metrics plus optional VLM scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from egojudge_metrics import edge_f1, hist_wasserstein, psnr, ssim_gray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Ground-truth ego frame.")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_to(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def near_black_fraction(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray < 12))


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = load_rgb(args.gt)
    rows = []
    for cand in args.candidates:
        p = Path(cand)
        img_raw = load_rgb(p)
        img = resize_to(img_raw, gt.shape[:2])
        row = {
            "name": p.name,
            "path": str(p.resolve()),
            "psnr": psnr(gt, img),
            "ssim": ssim_gray(gt, img),
            "edge_f1": edge_f1(gt, img),
            "rgb_hist_wasserstein": hist_wasserstein(gt, img),
            "near_black_fraction": near_black_fraction(img_raw),
        }
        hist_score = max(0.0, 1.0 - row["rgb_hist_wasserstein"] / 80.0)
        artifact_penalty = min(0.35, row["near_black_fraction"])
        row["selection_score"] = 100.0 * (
            0.45 * row["ssim"] + 0.20 * row["edge_f1"] + 0.25 * hist_score + 0.10 * min(row["psnr"] / 30.0, 1.0) - artifact_penalty
        )
        rows.append(row)
    rows.sort(key=lambda x: x["selection_score"], reverse=True)
    serializable = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in rows]
    (out / "variant_ranking.json").write_text(json.dumps({"ranking": serializable}, indent=2))
    lines = [
        "# Variant Ranking",
        "",
        "|rank|name|score|PSNR|SSIM|edge F1|RGB hist W|black frac|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(serializable, start=1):
        lines.append(
            f"|{idx}|{row['name']}|{row['selection_score']}|{row['psnr']}|{row['ssim']}|{row['edge_f1']}|{row['rgb_hist_wasserstein']}|{row['near_black_fraction']}|"
        )
    (out / "variant_ranking.md").write_text("\n".join(lines))
    print(f"Best: {serializable[0]['name'] if serializable else 'none'}")
    print(f"Wrote {out / 'variant_ranking.md'}")


if __name__ == "__main__":
    main()
