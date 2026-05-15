"""Interactive 3D point-cloud visualization of the exo→ego reprojection.

This is what answers the user's question: "why haven't i actually seen a 3D
point cloud visualization of the reprojection?"

Given one exo frame, we run monocular depth (Depth Anything V2), back-project
into the exo camera frame to get a coloured point cloud, then place a virtual
ego camera (eye position + look-at target) using the same head-anchor logic as
scripts/egoworld_pathway.py. The output is a self-contained HTML file (plotly)
that you can open and rotate, with two cameras drawn as frustums:
  - the exo camera at the origin
  - the virtual ego camera placed at the wearer's head, aimed at the workspace

Also writes a 2D PNG snapshot for the report figure.

Usage:
    python scripts/viz_3d_reprojection.py \
        --exo experiments/new_upload_sync/take2/frames_full/exo/exo_009_130.324.jpg \
        --out experiments/new_upload_sync/take2/viz_3d/exo_009 \
        [--max-points 60000]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def intrinsics_from_fov(W: int, H: int, hfov_deg: float):
    fx = W / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    return np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1.0]])


def backproject(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = depth_m.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_m.astype(np.float64)
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[0, 0]
    return np.stack([x, y, z], axis=-1)


def run_depth(exo_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb HxWx3 uint8, depth_m HxW float32)."""
    from transformers import pipeline as hf_pipeline
    rgb = np.array(Image.open(exo_path).convert("RGB"))
    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
    )
    out = depth_pipe(Image.fromarray(rgb))
    d = np.array(out["depth"]).astype(np.float32)
    d = d / (d.max() + 1e-6)
    depth_m = 0.3 + (1.0 - d) * (3.0 - 0.3)
    return rgb, depth_m


def detect_head_and_hands(exo_path: str, W: int, H: int):
    """Use MediaPipe Pose+Hands. Returns (head_uv or None, hand_centroid_uv_z or None)."""
    try:
        import mediapipe as mp
    except Exception:
        return None, None
    rgb = np.array(Image.open(exo_path).convert("RGB"))
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2,
                                  min_detection_confidence=0.2)
    head_uv = None
    pr = pose.process(rgb)
    if pr.pose_landmarks:
        lm = pr.pose_landmarks.landmark
        nose = lm[0]; le = lm[2]; re_ = lm[5]
        if min(le.visibility, re_.visibility) > 0.5:
            head_uv = ((le.x + re_.x) / 2.0, (le.y + re_.y) / 2.0)
        elif nose.visibility > 0.3:
            head_uv = (nose.x, max(0.0, nose.y - 0.02))
    hands_model = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                           model_complexity=1, min_detection_confidence=0.2)
    hr = hands_model.process(rgb)
    hand_uv = None
    if hr.multi_hand_landmarks:
        pts = []
        for h in hr.multi_hand_landmarks:
            for lm in h.landmark:
                pts.append((lm.x, lm.y))
        if pts:
            arr = np.array(pts)
            hand_uv = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    return head_uv, hand_uv


def frustum_lines(eye, target, up, hfov_deg, near=0.1, far=0.6, aspect=16/9):
    """Return a list of (x,y,z) line endpoints to draw a camera frustum."""
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-12)
    s = np.cross(up, f); s /= (np.linalg.norm(s) + 1e-12)
    u = np.cross(f, s)
    half_h = np.tan(np.deg2rad(hfov_deg) / 2.0)
    half_w = half_h * aspect
    def corner(d, hx, hy):
        return eye + f * d + s * (hx * d * half_w) + u * (hy * d * half_h)
    pts = {
        "n_tl": corner(near, -1, -1), "n_tr": corner(near, 1, -1),
        "n_br": corner(near, 1, 1),   "n_bl": corner(near, -1, 1),
        "f_tl": corner(far, -1, -1),  "f_tr": corner(far, 1, -1),
        "f_br": corner(far, 1, 1),    "f_bl": corner(far, -1, 1),
    }
    edges = [
        ("n_tl", "n_tr"), ("n_tr", "n_br"), ("n_br", "n_bl"), ("n_bl", "n_tl"),
        ("f_tl", "f_tr"), ("f_tr", "f_br"), ("f_br", "f_bl"), ("f_bl", "f_tl"),
        ("n_tl", "f_tl"), ("n_tr", "f_tr"), ("n_br", "f_br"), ("n_bl", "f_bl"),
        ("eye", "f_tl"), ("eye", "f_tr"), ("eye", "f_br"), ("eye", "f_bl"),
    ]
    pts["eye"] = eye
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [pts[a][0], pts[b][0], None]
        ys += [pts[a][1], pts[b][1], None]
        zs += [pts[a][2], pts[b][2], None]
    return xs, ys, zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=60000)
    ap.add_argument("--hfov-exo", type=float, default=80.0)
    ap.add_argument("--hfov-ego", type=float, default=100.0)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    rgb, depth_m = run_depth(args.exo)
    H, W, _ = rgb.shape
    K_exo = intrinsics_from_fov(W, H, args.hfov_exo)
    pts = backproject(depth_m, K_exo)

    # Subsample for browser performance
    n = pts.reshape(-1, 3).shape[0]
    stride = max(1, n // args.max_points)
    flat_pts = pts.reshape(-1, 3)[::stride]
    flat_rgb = rgb.reshape(-1, 3)[::stride]

    head_uv, hand_uv = detect_head_and_hands(args.exo, W, H)
    head_3d = None
    hand_3d = None
    if head_uv is not None:
        hx, hy = int(head_uv[0] * W), int(head_uv[1] * H)
        zh = float(depth_m[hy, hx])
        head_3d = np.array([(hx - K_exo[0, 2]) * zh / K_exo[0, 0],
                            (hy - K_exo[1, 2]) * zh / K_exo[0, 0], zh])
    if hand_uv is not None:
        hx, hy = int(hand_uv[0] * W), int(hand_uv[1] * H)
        zh = float(depth_m[hy, hx])
        hand_3d = np.array([(hx - K_exo[0, 2]) * zh / K_exo[0, 0],
                            (hy - K_exo[1, 2]) * zh / K_exo[0, 0], zh])

    # Virtual ego camera: at head, aimed at hands (with corrected -Z direction
    # for the wearer-facing-camera case)
    if head_3d is not None:
        eye_ego = head_3d.copy()
        if hand_3d is not None:
            target_ego = hand_3d.copy()
        else:
            target_ego = head_3d + np.array([0.0, 0.3, -0.5])
    else:
        eye_ego = np.array([0.0, -0.2, 1.8])
        target_ego = np.array([0.0, 0.2, 0.3])

    # Exo camera frustum at origin looking +Z
    exo_eye = np.array([0.0, 0.0, 0.0])
    exo_target = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, -1.0, 0.0])

    import plotly.graph_objects as go
    fig = go.Figure()

    # Point cloud
    fig.add_trace(go.Scatter3d(
        x=flat_pts[:, 0], y=flat_pts[:, 1], z=flat_pts[:, 2],
        mode="markers",
        marker=dict(size=1.6,
                    color=[f"rgb({r},{g},{b})" for r, g, b in flat_rgb.astype(int)],
                    opacity=0.85),
        name="Exo back-projected point cloud",
        hoverinfo="skip",
    ))

    # Exo frustum
    xs, ys, zs = frustum_lines(exo_eye, exo_target, up, args.hfov_exo, near=0.1, far=0.6)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                               line=dict(color="#1f77b4", width=4),
                               name="Exo camera"))

    # Virtual ego frustum
    xs, ys, zs = frustum_lines(eye_ego, target_ego, up, args.hfov_ego, near=0.05, far=0.4)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                               line=dict(color="#d62728", width=4),
                               name="Virtual ego camera"))

    # Head marker
    if head_3d is not None:
        fig.add_trace(go.Scatter3d(x=[head_3d[0]], y=[head_3d[1]], z=[head_3d[2]],
                                   mode="markers+text", marker=dict(size=6, color="#d62728"),
                                   text=["head"], textposition="top center", name="head"))
    if hand_3d is not None:
        fig.add_trace(go.Scatter3d(x=[hand_3d[0]], y=[hand_3d[1]], z=[hand_3d[2]],
                                   mode="markers+text", marker=dict(size=6, color="#2ca02c"),
                                   text=["hands"], textposition="bottom center", name="hand centroid"))

    fig.update_layout(
        title=f"Exo→Ego reprojection (3D) — {Path(args.exo).name}",
        scene=dict(
            xaxis_title="X (right)", yaxis_title="Y (down)", zaxis_title="Z (depth)",
            aspectmode="data",
            yaxis=dict(autorange="reversed"),  # image y is down
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.01, y=0.98),
    )

    html_path = out_dir / "pointcloud_3d.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    # static snapshot for the paper figure — matplotlib avoids the kaleido/Chrome dep
    png_path = out_dir / "pointcloud_3d.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig_m = plt.figure(figsize=(14, 9))
        ax = fig_m.add_subplot(111, projection="3d")
        ax.scatter(flat_pts[:, 0], -flat_pts[:, 1], flat_pts[:, 2],
                   c=flat_rgb / 255.0, s=0.3, alpha=0.85, marker=".")
        for label, eye, target, color in [
            ("Exo cam",          exo_eye,  exo_target,  "#1f77b4"),
            ("Virtual ego cam",  eye_ego,  target_ego,  "#d62728"),
        ]:
            ax.scatter(eye[0], -eye[1], eye[2], c=color, s=80, marker="^",
                       depthshade=False, label=label)
            ax.plot([eye[0], target[0]], [-eye[1], -target[1]], [eye[2], target[2]],
                    color=color, linewidth=2)
        if head_3d is not None:
            ax.scatter(head_3d[0], -head_3d[1], head_3d[2], c="#d62728", s=60,
                       marker="o", edgecolor="white", linewidth=1, label="head", depthshade=False)
        if hand_3d is not None:
            ax.scatter(hand_3d[0], -hand_3d[1], hand_3d[2], c="#2ca02c", s=60,
                       marker="o", edgecolor="white", linewidth=1, label="hands", depthshade=False)
        ax.set_xlabel("X (right)")
        ax.set_ylabel("Y (up = -image y)")
        ax.set_zlabel("Z (depth)")
        ax.set_title(f"Exo back-projected point cloud + virtual ego camera\n{Path(args.exo).name}")
        ax.legend(loc="upper left")
        ax.view_init(elev=18, azim=-55)
        plt.tight_layout()
        plt.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close(fig_m)
    except Exception as e:
        png_path = None
        print(f"(matplotlib snapshot skipped: {e})")

    meta = {
        "exo": args.exo,
        "n_points_total": int(n),
        "n_points_plotted": int(flat_pts.shape[0]),
        "head_uv": head_uv,
        "hand_uv": hand_uv,
        "head_3d": head_3d.tolist() if head_3d is not None else None,
        "hand_3d": hand_3d.tolist() if hand_3d is not None else None,
        "exo_camera": {"eye": exo_eye.tolist(), "target": exo_target.tolist(),
                       "hfov_deg": args.hfov_exo},
        "virtual_ego_camera": {"eye": eye_ego.tolist(), "target": target_ego.tolist(),
                               "hfov_deg": args.hfov_ego,
                               "note": "Eye is at head_3d; target is the hand centroid "
                                       "(or head + (0,0.3,-0.5) when no hands detected). "
                                       "-Z direction is correct for wearer-facing-camera setup."},
        "html": str(html_path),
        "png": str(png_path) if png_path else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {html_path}")
    print(f"plotted {flat_pts.shape[0]:,} / {n:,} points")


if __name__ == "__main__":
    main()
