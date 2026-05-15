#!/usr/bin/env python3
"""Create a report-friendly point-cloud/reprojection visualization for one frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from egoworld_pathway import backproject, build_sparse_ego, intrinsics_from_fov, run_perception


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-points", type=int, default=22000)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    perc = run_perception(args.exo)
    rgb = perc["rgb"]
    depth = perc["depth_m"]
    h, w = depth.shape
    k = intrinsics_from_fov(w, h, 80.0)
    pc = backproject(depth, k)

    sparse, hole = build_sparse_ego(perc)
    cv2.imwrite(str(out / "sparse_ego_rgb.png"), cv2.cvtColor(sparse, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "hole_mask.png"), hole)

    valid = np.isfinite(depth) & (depth > 0)
    ys, xs = np.where(valid)
    if len(xs) > args.max_points:
        idx = np.linspace(0, len(xs) - 1, args.max_points).astype(int)
        ys, xs = ys[idx], xs[idx]
    pts = pc[ys, xs]
    cols = rgb[ys, xs] / 255.0

    depth_vis = depth.copy()
    lo, hi = np.percentile(depth_vis[valid], [2, 98])
    depth_norm = np.clip((depth_vis - lo) / max(hi - lo, 1e-6), 0, 1)

    sparse_display = sparse.copy()
    sparse_display[hole > 0] = np.array([10, 12, 20], dtype=np.uint8)
    filled = 100.0 * float(np.count_nonzero(hole == 0)) / hole.size

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title("1. exo RGB input")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(depth_norm, cmap="magma_r")
    ax2.set_title("2. monocular relative depth")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    ax3.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=cols, s=0.35, alpha=0.75)
    ax3.set_title("3. exo point cloud")
    ax3.set_xlabel("x")
    ax3.set_ylabel("z")
    ax3.set_zlabel("-y")
    ax3.view_init(elev=18, azim=-62)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(sparse_display)
    ax4.set_title(f"4. virtual ego reprojection ({filled:.1f}% filled)")
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(hole, cmap="gray")
    ax5.set_title("5. hole mask for inpainting")
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(
        0,
        0.95,
        "Why this matters:\n"
        "- The point cloud is real structure from the exo image.\n"
        "- The virtual ego camera is still guessed from wearer/hand cues.\n"
        "- Holes reveal what geometry cannot see.\n"
        "- Generation must fill holes, but EgoJudge checks it did not break objects/contact.",
        va="top",
        fontsize=12,
    )
    ax6.axis("off")

    fig.tight_layout()
    fig.savefig(out / "pointcloud_reprojection_demo.png", dpi=170)
    plt.close(fig)

    meta = {
        "exo": args.exo,
        "filled_pct": round(filled, 3),
        "head_anchor_uv": perc.get("head_anchor_uv"),
        "n_hand_anchors": len(perc.get("hand_landmarks_3d", [])),
    }
    (out / "pointcloud_reprojection_demo.json").write_text(json.dumps(meta, indent=2))
    print(out / "pointcloud_reprojection_demo.png")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
