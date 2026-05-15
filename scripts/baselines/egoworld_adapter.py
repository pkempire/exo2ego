#!/usr/bin/env python3
"""Create an EgoWorld-style sparse/pose/text/dense record from local outputs.

EgoWorld's released loader expects JSONL records with paths:
  {"sparse": "...", "dense": "...", "pose": "...", "text": "..."}

This adapter does not run EgoWorld. It prepares our data in the same shape so
we can compare against or plug into their pipeline once checkpoints are set up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Source exo image.")
    parser.add_argument("--phase2", required=True, help="Phase2 JSON with hand detections.")
    parser.add_argument("--sparse", required=True, help="Sparse/geometry baseline image.")
    parser.add_argument("--dense", default=None, help="Reference/pseudo-GT ego image.")
    parser.add_argument("--text", required=True, help="Text prompt file.")
    parser.add_argument("--out", required=True, help="Output directory.")
    return parser.parse_args()


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_pose_map(phase2_path: str, out_shape: tuple[int, int]) -> np.ndarray:
    h, w = out_shape
    pose = np.zeros((h, w, 3), dtype=np.uint8)
    data = json.loads(Path(phase2_path).read_text())
    src_h, src_w = data.get("image_shape", [h, w])[:2]
    wrists = []
    for hand in data.get("hands", []):
        wx, wy = hand["wrist_pixel"]
        # Map exo wrist x roughly into ego image x. Put wrists in lower half,
        # because EgoWorld expects a hand-pose control map, not exact exo pixels.
        x = int(np.clip(wx / src_w * w, 0, w - 1))
        y = int(0.70 * h)
        wrists.append((x, y, hand.get("handedness", "")))

    for x, y, handedness in wrists:
        color = (80, 220, 255) if handedness.lower().startswith("left") else (255, 140, 80)
        cv2.circle(pose, (x, y), max(8, w // 80), color, -1)
        cv2.line(pose, (x, y), (x, h - 1), color, max(3, w // 220))
    if len(wrists) == 2:
        cv2.line(pose, wrists[0][:2], wrists[1][:2], (80, 255, 120), max(2, w // 300))
    return pose


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    sparse_dir = out / "sparse" / "demo"
    dense_dir = out / "dense" / "demo"
    pose_dir = out / "pose" / "demo"
    text_dir = out / "text" / "demo"
    data_dir = out / "data" / "demo" / "inpainting"
    for d in [sparse_dir, dense_dir, pose_dir, text_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    sparse = load_rgb(args.sparse)
    cv2.imwrite(str(sparse_dir / "000000.png"), cv2.cvtColor(sparse, cv2.COLOR_RGB2BGR))

    if args.dense:
        dense = load_rgb(args.dense)
    else:
        dense = sparse
    cv2.imwrite(str(dense_dir / "000000.png"), cv2.cvtColor(dense, cv2.COLOR_RGB2BGR))

    pose = make_pose_map(args.phase2, sparse.shape[:2])
    cv2.imwrite(str(pose_dir / "000000.png"), cv2.cvtColor(pose, cv2.COLOR_RGB2BGR))

    prompt = Path(args.text).read_text().strip()
    (text_dir / "000000.txt").write_text(prompt + "\n")

    record = {
        "sparse": "sparse/demo/000000.png",
        "dense": "dense/demo/000000.png",
        "pose": "pose/demo/000000.png",
        "text": "text/demo/000000.txt",
    }
    for split in ["train", "test"]:
        (data_dir / f"{split}.json").write_text(json.dumps(record) + "\n")

    print(f"Wrote EgoWorld-style adapter data to {out}")
    print(f"Record: {data_dir / 'test.json'}")


if __name__ == "__main__":
    main()
