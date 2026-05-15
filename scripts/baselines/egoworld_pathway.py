#!/usr/bin/env python3
"""EgoWorld-style sparse-RGB pathway (no learned ego-pose regressor).

Pipeline:
  exo image
    -> Depth Anything V2 (relative depth)
    -> MediaPipe Pose head landmark (where the wearer's head is in the exo)
    -> backproject exo pixels into 3D using assumed intrinsics + relative depth
    -> place a virtual ego camera at the detected head, looking forward+down
    -> project the 3D points into the ego camera (z-buffer)
    -> save sparse ego RGB + alpha hole mask
    -> send (sparse RGB, hole mask, text prompt) to OpenAI image-edit inpaint
       so the generator fills holes while preserving the projected pixels

This is intentionally cheap: no HaMeR (only MediaPipe Pose head), no learned
ego pose regressor, no SAM masks. The point is to test whether anchoring
the generator with a SPARSE RGB visual anchor beats text-only prompting.

Usage (per frame):
    python scripts/egoworld_pathway.py \\
        --exo path/to/exo.jpg \\
        --out experiments/egoworld_pathway/<name>/ \\
        [--prompt path/to/prompt.txt] \\
        [--skip-inpaint]   # produce sparse RGB only, no API call

Usage (batch over a manifest):
    python scripts/egoworld_pathway.py batch \\
        --plan experiments/egoexo4d_sfu_cooking025_7/pipeline_plan.json \\
        --out experiments/egoworld_pathway/cooking025_7/
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------- Geometry helpers ----------------

def intrinsics_from_fov(W: int, H: int, hfov_deg: float = 80.0) -> np.ndarray:
    """Reasonable intrinsics for a wide exo (cam01 = wide angle, hfov ~80-90 deg)."""
    fx = W / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    return np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float64)


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Backproject a HxW depth image into 3D points in the exo camera frame."""
    H, W = depth.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float64)
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[0, 0]
    return np.stack([x, y, z], axis=-1)  # H x W x 3


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a 4x4 view matrix (world -> camera) looking from eye to target."""
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    s = np.cross(up, f)
    s /= np.linalg.norm(s) + 1e-12
    u = np.cross(f, s)
    M = np.eye(4)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def project_points(
    pts_world: np.ndarray,
    rgb: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    Kego: np.ndarray,
    out_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Z-buffer projection of (H,W,3) points + (H,W,3) RGB into a virtual ego camera.

    Returns (sparse_rgb, hole_mask) where hole_mask is uint8 (255=hole, 0=filled).
    """
    Wout, Hout = out_size
    up = np.array([0.0, -1.0, 0.0])  # y points down in image; up vector in world
    V = look_at(eye, target, up)
    Hsrc, Wsrc, _ = pts_world.shape
    flat = pts_world.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)
    homog = np.hstack([flat, np.ones((flat.shape[0], 1))])
    cam = (V @ homog.T).T[:, :3]  # camera-frame points
    z = cam[:, 2]
    mask = z > 0.05
    cam = cam[mask]
    rgb_pts = rgb_flat[mask]
    u = Kego[0, 0] * cam[:, 0] / cam[:, 2] + Kego[0, 2]
    v = Kego[1, 1] * cam[:, 1] / cam[:, 2] + Kego[1, 2]
    in_bounds = (u >= 0) & (u < Wout) & (v >= 0) & (v < Hout)
    u = u[in_bounds].astype(np.int32)
    v = v[in_bounds].astype(np.int32)
    z = cam[in_bounds, 2]
    rgb_pts = rgb_pts[in_bounds]
    zbuf = np.full((Hout, Wout), np.inf, dtype=np.float64)
    sparse = np.zeros((Hout, Wout, 3), dtype=np.uint8)
    order = np.argsort(-z)  # paint far first
    for idx in order:
        x, y, zi = u[idx], v[idx], z[idx]
        if zi < zbuf[y, x]:
            zbuf[y, x] = zi
            sparse[y, x] = rgb_pts[idx]
    # splat 3x3 to make sparse pixels visible
    sparse_dilated = cv2.dilate(sparse, np.ones((3, 3), np.uint8))
    hole_mask = (sparse_dilated.sum(axis=-1) == 0).astype(np.uint8) * 255
    return sparse_dilated, hole_mask


# ---------------- Per-frame pipeline ----------------

def run_perception(exo_path: str) -> dict:
    """Run Depth Anything V2 + MediaPipe Pose + MediaPipe Hands. Returns a dict of arrays."""
    import mediapipe as mp
    from transformers import pipeline as hf_pipeline
    from PIL import Image

    bgr = cv2.imread(exo_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = rgb.shape

    # Depth
    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
    )
    out = depth_pipe(Image.fromarray(rgb))
    depth = np.array(out["depth"]).astype(np.float32)
    # Normalize to a plausible metric: foreground ~0.3 m, background ~3 m
    depth = depth / (depth.max() + 1e-6)
    depth_m = 0.3 + (1.0 - depth) * (3.0 - 0.3)

    # Pose
    pose_model = mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=2,
        min_detection_confidence=0.2, min_tracking_confidence=0.2,
    )
    pose_res = pose_model.process(rgb)
    head_anchor_uv: Optional[Tuple[float, float]] = None
    body_side: Optional[str] = None
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark
        nose = lm[0]; le = lm[2]; re = lm[5]
        if min(le.visibility, re.visibility) > 0.5:
            head_anchor_uv = ((le.x + re.x) / 2, (le.y + re.y) / 2)
        elif nose.visibility > 0.3:
            head_anchor_uv = (nose.x, max(0.0, nose.y - 0.02))
        sh_x = (lm[11].x + lm[12].x) / 2
        body_side = "left" if sh_x < 0.4 else "right" if sh_x > 0.6 else "center"

    # Hands: prefer MediaPipe HandLandmarker (21 landmarks), fall back to Pose wrist+elbow.
    hands_model = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=4, min_detection_confidence=0.2
    )
    hand_res = hands_model.process(rgb)
    hand_landmarks_3d = []
    if hand_res.multi_hand_landmarks:
        for hand in hand_res.multi_hand_landmarks:
            pts = []
            for lmk in hand.landmark:
                hx, hy = int(np.clip(lmk.x*W, 0, W-1)), int(np.clip(lmk.y*H, 0, H-1))
                pts.append((lmk.x, lmk.y, float(depth_m[hy, hx])))
            hand_landmarks_3d.append(pts)
    elif pose_res.pose_landmarks:
        # Fallback: Pose wrist+elbow gives us *some* arm anchor when HandLandmarker misses
        lm = pose_res.pose_landmarks.landmark
        for wrist_idx, elbow_idx in [(15, 13), (16, 14)]:
            wrist, elbow = lm[wrist_idx], lm[elbow_idx]
            if min(wrist.visibility, elbow.visibility) < 0.3:
                continue
            pts = []
            for lmk in (wrist, elbow):
                hx, hy = int(np.clip(lmk.x*W, 0, W-1)), int(np.clip(lmk.y*H, 0, H-1))
                pts.append((lmk.x, lmk.y, float(depth_m[hy, hx])))
            hand_landmarks_3d.append(pts)

    return {
        "rgb": rgb,
        "depth_m": depth_m,
        "head_anchor_uv": head_anchor_uv,
        "body_side": body_side,
        "hand_landmarks_3d": hand_landmarks_3d,
        "shape": (H, W),
    }


def build_sparse_ego(perc: dict, hfov_exo: float = 80.0, hfov_ego: float = 100.0,
                     out_size: Tuple[int, int] = (1024, 1024)) -> Tuple[np.ndarray, np.ndarray]:
    H, W = perc["shape"]
    K_exo = intrinsics_from_fov(W, H, hfov_exo)
    pts_world = backproject(perc["depth_m"], K_exo)
    head_uv = perc["head_anchor_uv"]

    # Compute hand-anchor 3D first so we can aim the virtual ego camera at them.
    # Exo coords: +X right, +Y down, +Z away from the exo camera.
    hand_anchor_world = []
    for hand_pts in perc.get("hand_landmarks_3d", []):
        for u_norm, v_norm, z_m in hand_pts:
            hx, hy = u_norm * W, v_norm * H
            x = (hx - K_exo[0, 2]) * z_m / K_exo[0, 0]
            y = (hy - K_exo[1, 2]) * z_m / K_exo[0, 0]
            hand_anchor_world.append(np.array([x, y, z_m]))

    if head_uv is None:
        # No head detected. Use mean hand 3D if available, else a sensible default
        # for "wearer is facing the exo camera". In that case forward = -Z (toward camera).
        if hand_anchor_world:
            target = np.mean(hand_anchor_world, axis=0)
            eye = target + np.array([0.0, -0.4, 0.6])   # head is above hands and farther from camera
        else:
            eye = np.array([0.0, -0.2, 1.8])
            target = np.array([0.0, 0.2, 0.3])           # look toward exo camera and down
    else:
        hx, hy = int(head_uv[0] * W), int(head_uv[1] * H)
        head_depth = float(perc["depth_m"][hy, hx])
        head_3d = np.array([
            (hx - K_exo[0, 2]) * head_depth / K_exo[0, 0],
            (hy - K_exo[1, 2]) * head_depth / K_exo[0, 0],
            head_depth,
        ])
        eye = head_3d.copy()
        if hand_anchor_world:
            # Aim at the hand centroid — this is where the wearer is actually looking.
            target = np.mean(hand_anchor_world, axis=0)
        else:
            # Fallback: wearer faces exo camera, so look toward camera (-Z) and down (+Y).
            target = head_3d + np.array([0.0, 0.3, -0.5])

    Kego = intrinsics_from_fov(out_size[0], out_size[1], hfov_ego)
    sparse, hole = project_points(pts_world, perc["rgb"], eye, target, Kego, out_size)
    if hand_anchor_world:
        up = np.array([0.0, -1.0, 0.0])
        V = look_at(eye, target, up)
        for p_world in hand_anchor_world:
            homog = np.array([*p_world, 1.0])
            cam = (V @ homog)[:3]
            if cam[2] > 0.05:
                u = Kego[0, 0] * cam[0] / cam[2] + Kego[0, 2]
                v = Kego[1, 1] * cam[1] / cam[2] + Kego[1, 2]
                if 0 <= u < out_size[0] and 0 <= v < out_size[1]:
                    # Draw a clear skin-toned dot for the hand anchor
                    cv2.circle(sparse, (int(u), int(v)), 8, (200, 165, 130), -1)
                    cv2.circle(hole, (int(u), int(v)), 8, 0, -1)
    return sparse, hole


def openai_inpaint(sparse_path: Path, hole_mask_path: Path, prompt: str, out_path: Path) -> None:
    """Send (sparse RGB, hole mask) to OpenAI images.edit inpaint mode."""
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set.")
    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("EGOJUDGE_IMAGE_MODEL", "gpt-image-1")
    with open(sparse_path, "rb") as fimg, open(hole_mask_path, "rb") as fmask:
        try:
            rsp = client.images.edit(
                model=model, image=fimg, mask=fmask, prompt=prompt,
                size="1024x1024", quality="medium", input_fidelity="high",
            )
        except TypeError:
            with open(sparse_path, "rb") as fimg2, open(hole_mask_path, "rb") as fmask2:
                rsp = client.images.edit(
                    model=model, image=fimg2, mask=fmask2, prompt=prompt,
                    size="1024x1024", quality="medium",
                )
    out_path.write_bytes(base64.b64decode(rsp.data[0].b64_json))


# ---------------- entry ----------------

def run_single(exo: str, out_dir: Path, prompt: Optional[str], skip_inpaint: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    perc = run_perception(exo)
    sparse, hole = build_sparse_ego(perc)

    # Visualizations
    sparse_path = out_dir / "sparse_ego_rgb.png"
    cv2.imwrite(str(sparse_path), cv2.cvtColor(sparse, cv2.COLOR_RGB2BGR))
    # Mask in OpenAI's expected convention: TRANSPARENT pixels = inpaint here.
    # We need to emit an RGBA PNG with alpha=0 where hole==255 (holes to fill).
    rgba = np.zeros((sparse.shape[0], sparse.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = cv2.cvtColor(sparse, cv2.COLOR_RGB2BGR)  # any color, ignored where alpha=0
    rgba[..., 3] = np.where(hole == 255, 0, 255).astype(np.uint8)
    mask_path = out_dir / "hole_mask_rgba.png"
    cv2.imwrite(str(mask_path), rgba)

    # Also save raw depth + head viz
    dn = ((perc["depth_m"] - perc["depth_m"].min())
          / (perc["depth_m"].max() - perc["depth_m"].min() + 1e-6) * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "depth.png"), cv2.applyColorMap(dn, cv2.COLORMAP_TURBO))

    meta = {
        "exo": exo,
        "head_anchor_uv": perc["head_anchor_uv"],
        "body_side": perc["body_side"],
        "sparse": str(sparse_path),
        "hole_mask_rgba": str(mask_path),
        "hole_fraction": float((hole == 255).mean()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if skip_inpaint:
        print(f"sparse RGB built: {sparse_path}, hole fraction: {meta['hole_fraction']:.2%}")
        return

    if prompt is None:
        prompt = (
            "This is a sparse egocentric (head-mounted) view reprojected from a "
            "third-person camera; transparent pixels are missing and must be "
            "inpainted to produce a complete photorealistic first-person view. "
            "Preserve every visible (non-transparent) pixel exactly as it is. "
            "Fill the transparent regions plausibly so the result looks like a "
            "real GoPro / Aria glasses ego frame: same scene, same lighting, "
            "natural arms entering from the bottom of the frame if appropriate."
        )
    out_path = out_dir / "ego_inpainted.png"
    openai_inpaint(sparse_path, mask_path, prompt, out_path)
    print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    p.add_argument("--exo")
    p.add_argument("--out")
    p.add_argument("--prompt", default=None)
    p.add_argument("--skip-inpaint", action="store_true")
    p1 = sub.add_parser("batch")
    p1.add_argument("--plan", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--skip-inpaint", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "batch":
        plan = json.loads(Path(args.plan).read_text())
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        for job in plan["jobs"]:
            run_single(job["exo"], out / f"frame_{job['idx']:03d}", None, args.skip_inpaint)
    else:
        if not args.exo or not args.out:
            print("usage: --exo PATH --out DIR  [--skip-inpaint]"); sys.exit(2)
        prompt = Path(args.prompt).read_text() if args.prompt else None
        run_single(args.exo, Path(args.out), prompt, args.skip_inpaint)


if __name__ == "__main__":
    main()
