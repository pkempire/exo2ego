#!/usr/bin/env python3
"""Estimate rough exo/ego view overlap from paired frames.

This is not full camera calibration. It uses feature matching + homography to
diagnose how much the exo and ego frames overlap on approximately planar
texture (e.g., a table cover). It helps answer: "where is the ego camera view
inside the exo image?" for paired data.
"""

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
    parser.add_argument("--exo", required=True)
    parser.add_argument("--ego", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exo = load_bgr(args.exo)
    ego = load_bgr(args.ego)
    exo_g = cv2.cvtColor(exo, cv2.COLOR_BGR2GRAY)
    ego_g = cv2.cvtColor(ego, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2500)
    kp_exo, des_exo = orb.detectAndCompute(exo_g, None)
    kp_ego, des_ego = orb.detectAndCompute(ego_g, None)
    report = {
        "exo_keypoints": len(kp_exo),
        "ego_keypoints": len(kp_ego),
        "matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "homography_ego_to_exo": None,
        "interpretation": "",
    }
    if des_exo is None or des_ego is None or len(kp_exo) < 8 or len(kp_ego) < 8:
        report["interpretation"] = "not enough local features for homography"
        (out / "view_calibration_report.json").write_text(json.dumps(report, indent=2))
        raise SystemExit(report["interpretation"])

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_ego, des_exo, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    report["matches"] = len(good)

    if len(good) >= 8:
        pts_ego = np.float32([kp_ego[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_exo = np.float32([kp_exo[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts_ego, pts_exo, cv2.RANSAC, 5.0)
    else:
        H, mask = None, None

    match_vis = cv2.drawMatches(
        ego,
        kp_ego,
        exo,
        kp_exo,
        good[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out / "feature_matches.jpg"), match_vis)

    exo_overlay = exo.copy()
    if H is not None and mask is not None:
        inliers = int(mask.ravel().sum())
        report["inliers"] = inliers
        report["inlier_ratio"] = float(inliers / max(len(good), 1))
        report["homography_ego_to_exo"] = H.tolist()
        eh, ew = ego.shape[:2]
        corners = np.float32([[0, 0], [ew - 1, 0], [ew - 1, eh - 1], [0, eh - 1]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H).astype(np.int32)
        cv2.polylines(exo_overlay, [projected], True, (0, 255, 255), 4)
        report["projected_ego_corners_in_exo_px"] = projected.reshape(-1, 2).tolist()
        if report["inlier_ratio"] > 0.25 and inliers > 20:
            report["interpretation"] = "usable planar overlap estimate"
        else:
            report["interpretation"] = "weak homography; use as diagnostic only"
    else:
        report["interpretation"] = "homography failed"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(exo, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Exo input")
    axes[1].imshow(cv2.cvtColor(ego, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Ego GT")
    axes[2].imshow(cv2.cvtColor(exo_overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Projected ego footprint in exo")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "view_overlap_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (out / "view_calibration_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {out / 'view_overlap_diagnostic.png'}")


if __name__ == "__main__":
    main()
