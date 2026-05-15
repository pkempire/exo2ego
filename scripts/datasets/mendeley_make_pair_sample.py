#!/usr/bin/env python3
"""Extract a small paired ego/exo sample from the Mendeley hand-view dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ego-zip", required=True)
    parser.add_argument("--exo-zip", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--activity", default="Dice")
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--stride", type=int, default=5)
    return parser.parse_args()


def image_members(zf: zipfile.ZipFile, activity: str) -> dict[str, str]:
    suffix = f"/{activity}/"
    out = {}
    for name in zf.namelist():
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if suffix not in name:
            continue
        out[Path(name).stem] = name
    return out


def extract_member(zf: zipfile.ZipFile, member: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, dst.open("wb") as f:
        shutil.copyfileobj(src, f)


def load_rgb(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_montage(records: list[dict], out: Path) -> None:
    if not records:
        return
    n = min(len(records), 6)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]
    for row_axes, rec in zip(axes, records[:n]):
        row_axes[0].imshow(load_rgb(Path(rec["exo"])))
        row_axes[0].set_title(f"Exo {rec['activity']} {rec['frame_id']}")
        row_axes[0].axis("off")
        row_axes[1].imshow(load_rgb(Path(rec["ego"])))
        row_axes[1].set_title(f"Ego GT {rec['activity']} {rec['frame_id']}")
        row_axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs_dir = out / "pairs"
    records = []

    with zipfile.ZipFile(args.ego_zip) as ego_zf, zipfile.ZipFile(args.exo_zip) as exo_zf:
        ego = image_members(ego_zf, args.activity)
        exo = image_members(exo_zf, args.activity)
        common = sorted(set(ego) & set(exo))
        if not common:
            raise SystemExit(f"No common frame ids for activity {args.activity}")
        chosen = common[:: max(1, args.stride)][: args.max_pairs]
        for frame_id in chosen:
            exo_dst = pairs_dir / f"{args.activity.replace(' ', '_')}_{frame_id}_exo.jpg"
            ego_dst = pairs_dir / f"{args.activity.replace(' ', '_')}_{frame_id}_ego_gt.jpg"
            extract_member(exo_zf, exo[frame_id], exo_dst)
            extract_member(ego_zf, ego[frame_id], ego_dst)
            records.append(
                {
                    "dataset": "Mendeley egocentric/exocentric view hands",
                    "activity": args.activity,
                    "frame_id": frame_id,
                    "exo": str(exo_dst.resolve()),
                    "ego": str(ego_dst.resolve()),
                    "exo_member": exo[frame_id],
                    "ego_member": ego[frame_id],
                }
            )

    (out / "pairs.json").write_text(json.dumps({"pairs": records}, indent=2))
    make_montage(records, out / "montage.png")
    print(f"Wrote {len(records)} pairs to {out}")
    print(f"Wrote {out / 'pairs.json'}")
    print(f"Wrote {out / 'montage.png'}")


if __name__ == "__main__":
    main()
