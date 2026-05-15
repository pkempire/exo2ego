#!/usr/bin/env python3
"""Evaluate generated exo-to-ego images against paired ego ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

from egojudge_metrics import edge_f1, hist_wasserstein, psnr, ssim_gray

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="pairs.json from mendeley_make_pair_sample.py")
    parser.add_argument("--candidates", nargs="+", required=True, help="Generated candidates named with frame id.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--display-width", type=int, default=640)
    parser.add_argument("--display-height", type=int, default=384)
    return parser.parse_args()


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_to(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def letterbox(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    out_w, out_h = size
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_h, out_w, 3), 245, dtype=np.uint8)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def find_pair(frame_id: str, pairs: list[dict]) -> dict | None:
    for pair in pairs:
        if pair["frame_id"] in frame_id:
            return pair
    return None


def make_visual(rows: list[dict], out: Path, display_size: tuple[int, int]) -> None:
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = np.array([axes])
    for row_axes, row in zip(axes, rows):
        for ax, key, title in [
            (row_axes[0], "exo", "Exo input"),
            (row_axes[1], "candidate", "Generated ego"),
            (row_axes[2], "ego", "GT ego"),
        ]:
            ax.imshow(letterbox(load_rgb(row[key]), display_size))
            ax.set_title(title)
            ax.axis("off")
        row_axes[1].set_xlabel(
            f"candidate resized to GT for metrics: PSNR {row['psnr']:.2f} | SSIM {row['ssim']:.3f} | Edge F1 {row['edge_f1']:.3f}",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(Path(args.pairs).read_text())["pairs"]

    rows = []
    for cand_path in args.candidates:
        cand_p = Path(cand_path)
        pair = find_pair(cand_p.name, pairs)
        if pair is None:
            continue
        gt = load_rgb(pair["ego"])
        cand_raw = load_rgb(cand_p)
        cand = resize_to(cand_raw, gt.shape[:2])
        exo_shape = load_rgb(pair["exo"]).shape[:2]
        row = {
            "frame_id": pair["frame_id"],
            "exo": pair["exo"],
            "ego": pair["ego"],
            "candidate": str(cand_p.resolve()),
            "candidate_name": cand_p.name,
            "exo_shape_hw": list(exo_shape),
            "ego_shape_hw": list(gt.shape[:2]),
            "candidate_shape_hw": list(cand_raw.shape[:2]),
            "metric_alignment": "candidate image is resized to the GT ego frame height/width before PSNR/SSIM/edge/histogram metrics",
            "psnr": psnr(gt, cand),
            "ssim": ssim_gray(gt, cand),
            "edge_f1": edge_f1(gt, cand),
            "rgb_hist_wasserstein": hist_wasserstein(gt, cand),
        }
        rows.append(row)

    serializable = []
    for row in rows:
        serializable.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    (out / "paired_metrics.json").write_text(json.dumps({"results": serializable}, indent=2))

    lines = [
        "# Paired Exo-to-Ego Evaluation",
        "",
        "Metrics are computed after resizing the generated candidate to the ground-truth ego image dimensions. The visual report uses letterboxing so the three images are easier to compare without changing aspect ratio.",
        "",
        "|frame|candidate|PSNR|SSIM|edge F1|RGB hist W|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in serializable:
        lines.append(
            f"|{row['frame_id']}|{row['candidate_name']}|{row['psnr']}|{row['ssim']}|{row['edge_f1']}|{row['rgb_hist_wasserstein']}|"
        )
    (out / "paired_metrics.md").write_text("\n".join(lines))
    make_visual(rows, out / "paired_visual_report.png", (args.display_width, args.display_height))
    print(f"Wrote {out / 'paired_metrics.md'}")
    print(f"Wrote {out / 'paired_visual_report.png'}")


if __name__ == "__main__":
    main()
