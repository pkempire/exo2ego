#!/usr/bin/env python3
"""Create a side-by-side exo/ego GT video from paired frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def letterbox(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = rgb.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def label(rgb: np.ndarray, text: str) -> np.ndarray:
    out = rgb.copy()
    cv2.putText(out, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.putText(out, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
    return out


def main() -> None:
    args = parse_args()
    pairs = json.loads(Path(args.pairs).read_text())["pairs"]
    half_w = args.width // 2
    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {args.out}")
    for pair in pairs:
        exo = label(letterbox(load_rgb(pair["exo"]), half_w, args.height), f"Exo {pair['frame_id']}")
        ego = label(letterbox(load_rgb(pair["ego"]), args.width - half_w, args.height), f"Ego GT {pair['frame_id']}")
        frame = np.concatenate([exo, ego], axis=1)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
