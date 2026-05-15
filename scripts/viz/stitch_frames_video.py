#!/usr/bin/env python3
"""Stitch image frames into a constant-FPS mp4."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.frames if Path(p).exists()]
    if not paths:
        raise SystemExit("no frames to stitch")
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(paths[0])
    h, w = first.shape[:2]
    out_w = args.width or w
    out_h = args.height or h
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(img)
    writer.release()
    print(f"Wrote {args.out} from {len(paths)} frames at {args.fps} fps")


if __name__ == "__main__":
    main()
