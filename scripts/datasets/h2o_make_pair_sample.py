#!/usr/bin/env python3
"""Extract a tiny paired exo/ego sample from an unpacked H2O dataset.

The script is intentionally structure-tolerant: it searches for directories
containing cam0..cam4 with rgb folders, then copies synchronized frame names.
cam0..cam3 are treated as exo views and cam4 as ego ground truth.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Extracted H2O root directory.")
    parser.add_argument("--out", required=True, help="Output sample directory.")
    parser.add_argument("--max-frames", type=int, default=8)
    return parser.parse_args()


def rgb_dir(cam_dir: Path) -> Path | None:
    candidates = [cam_dir / "rgb", cam_dir / "color", cam_dir / "image"]
    for cand in candidates:
        if cand.is_dir():
            return cand
    image_files = [p for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return cam_dir if image_files else None


def frame_map(directory: Path) -> dict[str, Path]:
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return {p.stem: p for p in files}


def find_sequences(root: Path) -> list[dict]:
    sequences = []
    for cam4 in root.rglob("cam4"):
        parent = cam4.parent
        cams = {}
        ok = True
        for i in range(5):
            cdir = parent / f"cam{i}"
            rdir = rgb_dir(cdir) if cdir.is_dir() else None
            if rdir is None:
                ok = False
                break
            cams[f"cam{i}"] = rdir
        if ok:
            sequences.append({"sequence": parent, "cams": cams})
    return sequences


def read_rgb(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_montage(records: list[dict], out: Path) -> None:
    if not records:
        return
    first = records[0]
    labels = ["exo_cam0", "exo_cam1", "exo_cam2", "exo_cam3", "ego_cam4_gt"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, label in zip(axes, labels):
        ax.imshow(read_rgb(Path(first[label])))
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sequences = find_sequences(root)
    if not sequences:
        raise SystemExit(f"No H2O-like cam0..cam4 sequence found under {root}")

    chosen = None
    common_frames: list[str] = []
    maps = {}
    for seq in sequences:
        maps = {name: frame_map(path) for name, path in seq["cams"].items()}
        common = set.intersection(*(set(m.keys()) for m in maps.values()))
        if common:
            chosen = seq
            common_frames = sorted(common)
            break
    if chosen is None:
        raise SystemExit("Found cam directories but no synchronized frame names across cam0..cam4.")

    records = []
    for frame_id in common_frames[: args.max_frames]:
        rec = {"frame_id": frame_id, "sequence": str(chosen["sequence"].resolve())}
        for i in range(4):
            src = maps[f"cam{i}"][frame_id]
            dst = out / f"{frame_id}_exo_cam{i}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            rec[f"exo_cam{i}"] = str(dst.resolve())
        src = maps["cam4"][frame_id]
        dst = out / f"{frame_id}_ego_cam4_gt{src.suffix.lower()}"
        shutil.copy2(src, dst)
        rec["ego_cam4_gt"] = str(dst.resolve())
        records.append(rec)

    (out / "pairs.json").write_text(json.dumps({"sequence": str(chosen["sequence"].resolve()), "pairs": records}, indent=2))
    make_montage(records, out / "montage.png")
    print(f"Found sequence: {chosen['sequence']}")
    print(f"Wrote {len(records)} frame pairs to {out}")
    print(f"Wrote {out / 'pairs.json'}")
    print(f"Wrote {out / 'montage.png'}")


if __name__ == "__main__":
    main()
