#!/usr/bin/env python3
"""Extract paired ego/exo frames directly from the nested Mendeley archive.

The downloaded Mendeley file is a zip containing many inner zip files. This
script reads the needed ego/exo inner zips into memory and extracts only a small
paired sample, avoiding a full 8GB+ expansion.
"""

from __future__ import annotations

import argparse
import io
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
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--group", default="Two Persons")
    parser.add_argument("--lighting", default="Dim Light")
    parser.add_argument("--activity", default="Dice")
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--stride", type=int, default=80)
    parser.add_argument("--offset", type=int, default=0, help="Offset into the strided common frame list.")
    return parser.parse_args()


def inner_zip_name(view: str, group: str, lighting: str) -> str:
    return f"A dataset of egocentric and exocentric view hands in interactive senses/{view}/{group}/{lighting}.zip"


def open_inner_zip(outer: zipfile.ZipFile, member: str) -> zipfile.ZipFile:
    data = outer.read(member)
    return zipfile.ZipFile(io.BytesIO(data))


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
    fig, axes = plt.subplots(len(records), 2, figsize=(10, 3.2 * len(records)))
    if len(records) == 1:
        axes = [axes]
    for row_axes, rec in zip(axes, records):
        row_axes[0].imshow(load_rgb(Path(rec["exo"])))
        row_axes[0].set_title(f"Exo {rec['frame_id']}")
        row_axes[0].axis("off")
        row_axes[1].imshow(load_rgb(Path(rec["ego"])))
        row_axes[1].set_title(f"Ego GT {rec['frame_id']}")
        row_axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs_dir = out / "pairs"
    ego_member = inner_zip_name("Egocentric View", args.group, args.lighting)
    exo_member = inner_zip_name("Exocentric View", args.group, args.lighting)

    records = []
    with zipfile.ZipFile(archive) as outer:
        missing = [m for m in [ego_member, exo_member] if m not in outer.namelist()]
        if missing:
            raise SystemExit(f"Missing inner zips: {missing}")
        with open_inner_zip(outer, ego_member) as ego_zf, open_inner_zip(outer, exo_member) as exo_zf:
            ego = image_members(ego_zf, args.activity)
            exo = image_members(exo_zf, args.activity)
            common = sorted(set(ego) & set(exo))
            if not common:
                raise SystemExit(f"No common frame ids for {args.group}/{args.lighting}/{args.activity}")
            strided = common[:: max(1, args.stride)]
            chosen = strided[args.offset : args.offset + args.max_pairs]
            if not chosen:
                raise SystemExit("No frames selected; reduce offset/stride.")
            prefix = args.activity.replace(" ", "_")
            for frame_id in chosen:
                exo_dst = pairs_dir / f"{prefix}_{frame_id}_exo.jpg"
                ego_dst = pairs_dir / f"{prefix}_{frame_id}_ego_gt.jpg"
                extract_member(exo_zf, exo[frame_id], exo_dst)
                extract_member(ego_zf, ego[frame_id], ego_dst)
                records.append(
                    {
                        "dataset": "Mendeley egocentric/exocentric view hands",
                        "group": args.group,
                        "lighting": args.lighting,
                        "activity": args.activity,
                        "frame_id": frame_id,
                        "exo": str(exo_dst.resolve()),
                        "ego": str(ego_dst.resolve()),
                        "exo_member": exo[frame_id],
                        "ego_member": ego[frame_id],
                        "outer_archive": str(archive.resolve()),
                        "ego_inner_zip": ego_member,
                        "exo_inner_zip": exo_member,
                    }
                )

    (out / "pairs.json").write_text(json.dumps({"pairs": records}, indent=2))
    make_montage(records, out / "montage.png")
    print(f"Wrote {len(records)} pairs to {out}")
    print(f"Wrote {out / 'pairs.json'}")
    print(f"Wrote {out / 'montage.png'}")


if __name__ == "__main__":
    main()
