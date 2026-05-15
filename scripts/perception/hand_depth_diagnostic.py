#!/usr/bin/env python3
"""Run hand-landmark and depth diagnostics on exo/generated/GT images."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
    parser.add_argument("--skip-depth", action="store_true")
    return parser.parse_args()


def ensure_hand_model(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe hand landmarker model to {path}")
    urllib.request.urlretrieve(HAND_MODEL_URL, path)


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detect_hands(rgb: np.ndarray, model_path: Path) -> list[dict]:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base = python.BaseOptions(model_asset_path=str(model_path))
    opts = vision.HandLandmarkerOptions(base_options=base, num_hands=4)
    detector = vision.HandLandmarker.create_from_options(opts)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    detector.close()

    h, w = rgb.shape[:2]
    hands = []
    for idx, landmarks in enumerate(result.hand_landmarks):
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        handed = "unknown"
        if idx < len(result.handedness) and result.handedness[idx]:
            handed = result.handedness[idx][0].category_name
        hands.append(
            {
                "handedness": handed,
                "bbox_px": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "wrist_px": pts[0].tolist(),
                "landmarks_px": pts.tolist(),
            }
        )
    return hands


def depth_map(rgb: np.ndarray):
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else (-1 if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available() else "mps")
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    result = pipe(Image.fromarray(rgb))
    pred = result["predicted_depth"]
    arr = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.array(pred)
    arr = np.squeeze(arr).astype(np.float32)
    return arr


def draw_hand_overlay(rgb: np.ndarray, hands: list[dict]) -> np.ndarray:
    out = rgb.copy()
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    for hand in hands:
        pts = np.array(hand["landmarks_px"], dtype=np.int32)
        for a, b in connections:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), (40, 255, 80), 2)
        for p in pts:
            cv2.circle(out, tuple(p), 3, (255, 220, 40), -1)
        x, y, w, h = [int(v) for v in hand["bbox_px"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 220, 40), 2)
        cv2.putText(out, hand["handedness"], (x, max(16, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 40), 2)
    return out


def depth_vis(depth: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(depth[np.isfinite(depth)], [2, 98])
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    return (plt.cm.inferno(norm)[..., :3] * 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.names or [Path(p).stem for p in args.images]
    model_path = Path(args.hand_model)
    ensure_hand_model(model_path)

    report = {"images": []}
    fig, axes = plt.subplots(len(args.images), 3 if not args.skip_depth else 2, figsize=(15, 5 * len(args.images)))
    if len(args.images) == 1:
        axes = np.array([axes])
    for row, (path, name) in enumerate(zip(args.images, names)):
        rgb = load_rgb(path)
        hands = detect_hands(rgb, model_path)
        hand_overlay = draw_hand_overlay(rgb, hands)
        item = {
            "name": name,
            "path": str(Path(path).resolve()),
            "image_shape_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
            "hands_detected": len(hands),
            "hands": hands,
        }
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{name}: input")
        axes[row, 1].imshow(hand_overlay)
        axes[row, 1].set_title(f"{name}: hands ({len(hands)})")
        if not args.skip_depth:
            d = depth_map(rgb)
            item["depth_summary"] = {
                "min": float(np.nanmin(d)),
                "median": float(np.nanmedian(d)),
                "max": float(np.nanmax(d)),
            }
            axes[row, 2].imshow(depth_vis(d))
            axes[row, 2].set_title(f"{name}: monocular depth")
        for ax in axes[row]:
            ax.axis("off")
        report["images"].append(item)
    fig.tight_layout()
    fig.savefig(out / "hand_depth_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    (out / "hand_depth_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {out / 'hand_depth_diagnostic.png'}")
    print(f"Wrote {out / 'hand_depth_report.json'}")


if __name__ == "__main__":
    main()
