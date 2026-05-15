#!/usr/bin/env python3
"""Deterministic geometry-only exo-to-ego baseline.

This is the honest "standard pipeline" baseline:

  monocular depth + exo intrinsics -> point cloud
  approximate eye camera from detected hand location
  point-cloud reprojection -> sparse ego image + mask
  optional nonlearned CV2 inpaint -> weak dense baseline

It intentionally avoids generative models. The output should be compared
against EgoJudge/OpenAI/Gemini/EgoWorld-style outputs to show exactly where
geometry alone succeeds or fails when the true ego camera pose is unknown.
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
    parser.add_argument("--phase2", required=True, help="Phase2 JSON containing rgb, depth_map, hands.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--exo-hfov", type=float, default=55.0)
    parser.add_argument("--ego-hfov", type=float, default=110.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--target-hand-depth-m", type=float, default=0.75)
    parser.add_argument("--eye-above-hand-m", type=float, default=0.48)
    parser.add_argument("--eye-behind-hand-m", type=float, default=0.42)
    parser.add_argument("--look-down-m", type=float, default=0.08)
    parser.add_argument("--splat-radius", type=int, default=2)
    return parser.parse_args()


def estimate_intrinsics(w: int, h: int, hfov_deg: float) -> np.ndarray:
    fx = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def backproject(depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth
    x = (uu - k[0, 2]) * z / k[0, 0]
    y = (vv - k[1, 2]) * z / k[1, 1]
    return np.stack([x, y, z], axis=-1)


def robust_scale_depth(depth: np.ndarray, hands: list[dict], target_hand_depth_m: float) -> float:
    samples = []
    h, w = depth.shape
    for hand in hands:
        wx, wy = hand.get("wrist_pixel", [None, None])
        if wx is None or wy is None:
            continue
        if 0 <= wx < w and 0 <= wy < h:
            samples.append(float(depth[int(wy), int(wx)]))
    if not samples:
        return 1.0
    med = float(np.median(samples))
    if med <= 1e-6:
        return 1.0
    return target_hand_depth_m / med


def hand_center_3d(pc: np.ndarray, depth: np.ndarray, hands: list[dict]) -> np.ndarray:
    points = []
    h, w = depth.shape
    for hand in hands:
        wx, wy = hand.get("wrist_pixel", [None, None])
        if wx is None or wy is None:
            continue
        wx, wy = int(wx), int(wy)
        if 0 <= wx < w and 0 <= wy < h and np.isfinite(depth[wy, wx]):
            points.append(pc[wy, wx])
    if points:
        return np.mean(np.stack(points, axis=0), axis=0)
    valid = np.isfinite(depth) & (depth > 0)
    return np.median(pc[valid], axis=0)


def make_camera_basis(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Exo coordinates: +X image-right, +Y image-down, +Z away from exo camera.
    # Ego camera convention here: +Z forward, +X right, +Y down.
    z_axis = normalize(target - eye)
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    x_axis = normalize(np.cross(world_up, z_axis))
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    y_axis = normalize(np.cross(x_axis, z_axis))
    return x_axis, y_axis, z_axis


def project_sparse(
    rgb: np.ndarray,
    pc: np.ndarray,
    valid: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    out_w: int,
    out_h: int,
    ego_hfov: float,
    splat_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis, y_axis, z_axis = make_camera_basis(eye, target)
    pts = pc.reshape(-1, 3)
    cols = rgb.reshape(-1, 3)
    vmask = valid.reshape(-1)

    rel = pts[vmask] - eye[None, :]
    cam_x = rel @ x_axis
    cam_y = rel @ y_axis
    cam_z = rel @ z_axis
    cols = cols[vmask]

    in_front = cam_z > 0.03
    cam_x, cam_y, cam_z, cols = cam_x[in_front], cam_y[in_front], cam_z[in_front], cols[in_front]

    k_ego = estimate_intrinsics(out_w, out_h, ego_hfov)
    u = np.round(k_ego[0, 0] * cam_x / cam_z + k_ego[0, 2]).astype(np.int32)
    v = np.round(k_ego[1, 1] * cam_y / cam_z + k_ego[1, 2]).astype(np.int32)
    in_bounds = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    u, v, cam_z, cols = u[in_bounds], v[in_bounds], cam_z[in_bounds], cols[in_bounds]

    order = np.argsort(cam_z)[::-1]  # far first, near last overwrites.
    sparse = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    depth = np.full((out_h, out_w), np.inf, dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    r = max(0, int(splat_radius))
    for idx in order:
        x, y, z = int(u[idx]), int(v[idx]), float(cam_z[idx])
        x0, x1 = max(0, x - r), min(out_w, x + r + 1)
        y0, y1 = max(0, y - r), min(out_h, y + r + 1)
        patch = depth[y0:y1, x0:x1]
        nearer = z < patch
        patch[nearer] = z
        sparse[y0:y1, x0:x1][nearer] = cols[idx]
        mask[y0:y1, x0:x1][nearer] = 255
    return sparse, mask, depth


def save_depth(path: Path, depth: np.ndarray, mask: np.ndarray) -> None:
    valid = mask > 0
    vis = np.zeros_like(depth, dtype=np.uint8)
    if valid.any():
        vals = depth[valid]
        lo, hi = np.percentile(vals, [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        vis = (255 * (1.0 - norm)).astype(np.uint8)
        vis[~valid] = 0
    cv2.imwrite(str(path), vis)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.phase2).read_text())
    rgb = np.asarray(data["rgb"], dtype=np.uint8)
    depth = np.asarray(data["depth_map"], dtype=np.float32)
    hands = data.get("hands", [])
    h, w = depth.shape

    scale = robust_scale_depth(depth, hands, args.target_hand_depth_m)
    depth_m = depth * scale
    k_exo = estimate_intrinsics(w, h, args.exo_hfov)
    pc = backproject(depth_m, k_exo)
    valid = np.isfinite(depth_m) & (depth_m > 0.02) & (depth_m < 8.0)

    hand_center = hand_center_3d(pc, depth_m, hands)
    eye = hand_center + np.array([0.0, -args.eye_above_hand_m, args.eye_behind_hand_m], dtype=np.float32)
    target = hand_center + np.array([0.0, args.look_down_m, 0.0], dtype=np.float32)

    sparse, mask, ego_depth = project_sparse(
        rgb=rgb,
        pc=pc,
        valid=valid,
        eye=eye,
        target=target,
        out_w=args.width,
        out_h=args.height,
        ego_hfov=args.ego_hfov,
        splat_radius=args.splat_radius,
    )

    holes = 255 - mask
    inpaint = cv2.inpaint(cv2.cvtColor(sparse, cv2.COLOR_RGB2BGR), holes, 5, cv2.INPAINT_TELEA)
    inpaint = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)

    cv2.imwrite(str(out / "standard_sparse.png"), cv2.cvtColor(sparse, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "standard_mask.png"), mask)
    cv2.imwrite(str(out / "standard_inpaint_cv2.png"), cv2.cvtColor(inpaint, cv2.COLOR_RGB2BGR))
    save_depth(out / "standard_depth.png", ego_depth, mask)

    sparse_display = sparse.copy()
    sparse_display[mask == 0] = np.array([14, 15, 24], dtype=np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("Exo input")
    axes[1].imshow(sparse_display)
    axes[1].set_title(f"Sparse ego ({100.0 * np.count_nonzero(mask) / mask.size:.1f}% filled)")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Projected mask")
    axes[3].imshow(inpaint)
    axes[3].set_title("CV2 inpaint baseline")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out / "standard_baseline_debug.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = {
        "method": "deterministic_depth_pointcloud_virtual_eye",
        "filled_pct": round(100.0 * np.count_nonzero(mask) / mask.size, 3),
        "scale": float(scale),
        "hand_center_exo_m": hand_center.tolist(),
        "eye_exo_m": eye.tolist(),
        "target_exo_m": target.tolist(),
        "exo_hfov_deg": args.exo_hfov,
        "ego_hfov_deg": args.ego_hfov,
        "eye_above_hand_m": args.eye_above_hand_m,
        "eye_behind_hand_m": args.eye_behind_hand_m,
        "splat_radius_px": args.splat_radius,
    }
    (out / "standard_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
