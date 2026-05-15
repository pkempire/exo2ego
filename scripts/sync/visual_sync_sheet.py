#!/usr/bin/env python3
"""Build visual sheets for manual exo/ego synchronization.

This is deliberately simple: choose one visually identifiable exo anchor
frame, then render nearby exo frames and all/nearby ego candidates with large
frame labels. The user picks the ego frame that shows the same event, e.g.
"first touch of the book"; that anchor gives the integer frame offset used to
write synced pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/own_capture/take1")
    p.add_argument("--exo-anchor", type=int, required=True)
    p.add_argument("--ego-anchor", type=int, default=None)
    p.add_argument("--exo-window", type=int, default=6)
    p.add_argument("--ego-start", type=int, default=None)
    p.add_argument("--ego-end", type=int, default=None)
    p.add_argument("--cell-w", type=int, default=320)
    p.add_argument("--out", required=True)
    p.add_argument("--write-pairs", default=None)
    return p.parse_args()


def frame_path(root: Path, stream: str, idx: int) -> Path:
    return root / f"{stream}_frames" / f"{stream}_{idx:03d}.jpg"


def load_cell(path: Path, label: str, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    img = cv2.imread(str(path))
    if img is None:
        cell = np.full((h, w, 3), 35, np.uint8)
        cv2.putText(cell, f"missing {label}", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 240), 2)
        return cell
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    cell = np.full((h, w, 3), 245, np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    cell[y:y + nh, x:x + nw] = resized
    cv2.rectangle(cell, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(cell, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return cell


def make_sheet(root: Path, exo_anchor: int, exo_window: int, ego_indices: list[int], out: Path, cell_w: int) -> None:
    cell_h = int(cell_w * 9 / 16)
    exo_indices = list(range(max(1, exo_anchor - exo_window), exo_anchor + exo_window + 1))
    exo_cells = [
        load_cell(frame_path(root, "exo", i), f"exo_{i:03d}" + ("  ANCHOR" if i == exo_anchor else ""), (cell_w, cell_h))
        for i in exo_indices
    ]
    ego_cells = [load_cell(frame_path(root, "ego", i), f"ego_{i:03d}", (cell_w, cell_h)) for i in ego_indices]

    cols = 4
    rows = []
    for cells in (exo_cells, ego_cells):
        for i in range(0, len(cells), cols):
            row = cells[i:i + cols]
            while len(row) < cols:
                row.append(np.full((cell_h, cell_w, 3), 245, np.uint8))
            rows.append(np.concatenate(row, axis=1))
        rows.append(np.full((16, cols * cell_w, 3), 255, np.uint8))
    sheet = np.concatenate(rows, axis=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)


def write_pairs(root: Path, exo_anchor: int, ego_anchor: int, out: Path) -> None:
    delta = exo_anchor - ego_anchor
    exo_paths = sorted((root / "exo_frames").glob("exo_*.jpg"))
    ego_set = {int(p.stem.split("_")[-1]): p for p in (root / "ego_frames").glob("ego_*.jpg")}
    pairs = []
    for exo_path in exo_paths:
        exo_idx = int(exo_path.stem.split("_")[-1])
        ego_idx = exo_idx - delta
        ego_path = ego_set.get(ego_idx)
        if ego_path is None:
            continue
        pairs.append({
            "exo": str(exo_path),
            "ego": str(ego_path),
            "exo_idx": exo_idx,
            "ego_idx": ego_idx,
        })
    out.write_text(json.dumps({
        "pairs": pairs,
        "delta_exo_minus_ego": delta,
        "user_sync": f"exo#{exo_anchor} = ego#{ego_anchor} (manual visual anchor)",
    }, indent=2))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if args.ego_start is None or args.ego_end is None:
        ego_indices = [int(p.stem.split("_")[-1]) for p in sorted((root / "ego_frames").glob("ego_*.jpg"))]
    else:
        ego_indices = list(range(args.ego_start, args.ego_end + 1))
    make_sheet(root, args.exo_anchor, args.exo_window, ego_indices, Path(args.out), args.cell_w)
    print(f"wrote {args.out}")
    if args.write_pairs:
        if args.ego_anchor is None:
            raise SystemExit("--write-pairs requires --ego-anchor")
        write_pairs(root, args.exo_anchor, args.ego_anchor, Path(args.write_pairs))
        print(f"wrote {args.write_pairs}")


if __name__ == "__main__":
    main()
