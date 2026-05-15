#!/usr/bin/env python3
"""Measure simple temporal consistency for a sequence of generated frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from egojudge_metrics import hist_wasserstein, ssim_gray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    return parser.parse_args()


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (640, 384), interpolation=cv2.INTER_AREA)


def flow_warp_error(prev: np.ndarray, curr: np.ndarray) -> float:
    prev_g = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_g = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_g, curr_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = prev_g.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx + flow[..., 0]).astype(np.float32)
    map_y = (yy + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return float(np.mean(np.abs(warped.astype(np.float32) - curr.astype(np.float32))) / 255.0)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in args.frames]
    frames = [load_rgb(p) for p in paths]
    rows = []
    for i in range(1, len(frames)):
        rows.append(
            {
                "prev": paths[i - 1].name,
                "curr": paths[i].name,
                "ssim": round(ssim_gray(frames[i - 1], frames[i]), 4),
                "rgb_hist_wasserstein": round(hist_wasserstein(frames[i - 1], frames[i]), 4),
                "flow_warp_l1": round(flow_warp_error(frames[i - 1], frames[i]), 4),
            }
        )
    summary = {}
    if rows:
        for key in ["ssim", "rgb_hist_wasserstein", "flow_warp_l1"]:
            vals = [row[key] for row in rows]
            summary[f"mean_{key}"] = round(float(np.mean(vals)), 4)
    (out / "temporal_metrics.json").write_text(json.dumps({"summary": summary, "pairs": rows}, indent=2))
    lines = [
        "# Temporal Consistency",
        "",
        "Higher adjacent SSIM is smoother; lower histogram distance and lower flow-warp L1 are smoother. These metrics should be interpreted with caution because true object motion should still change pixels.",
        "",
        "|prev|curr|SSIM|RGB hist W|flow warp L1|",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"|{row['prev']}|{row['curr']}|{row['ssim']}|{row['rgb_hist_wasserstein']}|{row['flow_warp_l1']}|")
    (out / "temporal_metrics.md").write_text("\n".join(lines))
    print(f"Wrote {out / 'temporal_metrics.md'}")


if __name__ == "__main__":
    main()
