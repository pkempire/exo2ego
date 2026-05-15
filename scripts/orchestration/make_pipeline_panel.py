"""Build a single-frame side-by-side panel of every pipeline for the demo video.

Usage:
    python scripts/make_pipeline_panel.py \
        --frame exo_013 \
        --root data/own_capture/take1 \
        --out data/own_capture/take1/pipeline_panel_exo_013.png
"""
import argparse
import json
import os
from PIL import Image, ImageDraw, ImageFont


def lookup_ego_for_exo(pairs_path, exo_idx):
    """Use pairs.json to find the GT ego that corresponds to this exo.
    Returns the ego index (int) or None if no pair exists."""
    if not os.path.exists(pairs_path):
        return None
    pairs = json.load(open(pairs_path))["pairs"]
    for r in pairs:
        if r["exo_idx"] == exo_idx:
            return r["ego_idx"]
    return None


def load(path, target_size):
    if not os.path.exists(path):
        img = Image.new("RGB", target_size, color=(40, 40, 40))
        d = ImageDraw.Draw(img)
        d.text((20, 20), f"missing: {os.path.basename(path)}", fill=(220, 80, 80))
        return img
    img = Image.open(path).convert("RGB")
    img.thumbnail(target_size, Image.LANCZOS)
    canvas = Image.new("RGB", target_size, (0, 0, 0))
    x = (target_size[0] - img.size[0]) // 2
    y = (target_size[1] - img.size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="exo_013")
    ap.add_argument("--root", default="data/own_capture/take1")
    ap.add_argument("--out", default="data/own_capture/take1/pipeline_panel.png")
    ap.add_argument("--cell", type=int, default=640)
    args = ap.parse_args()

    frame_id = args.frame                        # e.g. exo_013
    exo_idx = int(frame_id.split("_")[-1])       # 13
    R = args.root
    suffix = "_focus" if frame_id == "exo_013" else ""

    # Resolve the correct GT ego frame using pairs.json (not naive matching index).
    ego_idx = lookup_ego_for_exo(f"{R}/pairs.json", exo_idx)
    if ego_idx is not None:
        gt_path = f"{R}/ego_frames/ego_{ego_idx:03d}.jpg"
        gt_label = f"Ground-truth ego (ego_{ego_idx:03d}, paired)"
    else:
        gt_path = f"{R}/ego_frames/ego_{exo_idx:03d}.jpg"
        gt_label = "Ground-truth ego (NO PAIR — naive index)"

    # Prefer the v2 sparse render (camera-direction fix). Fall back to v1 if missing.
    sparse_v2 = f"{R}/egoworld_sparse{suffix}_v2/{frame_id}/sparse_ego_rgb.png"
    sparse_v1 = f"{R}/egoworld_sparse{suffix}/{frame_id}/sparse_ego_rgb.png"
    sparse_path = sparse_v2 if os.path.exists(sparse_v2) else sparse_v1
    sparse_label = "Sparse ego (geometry, v2)" if sparse_path == sparse_v2 else "Sparse ego (geometry)"

    panels = [
        (f"Source exo ({frame_id})",   f"{R}/exo_frames/{frame_id}.jpg"),
        (sparse_label,                  sparse_path),
        ("Baseline T2I",                f"{R}/generated_baseline{suffix}/{frame_id}.png"),
        ("EgoWorld (geom-conditioned)", f"{R}/generated_egoworld{suffix}/{frame_id}.png"),
        ("Wearer-conditioned prompt",   f"{R}/generated_wearer{suffix}/{frame_id}.png"),
        (gt_label,                      gt_path),
    ]

    cell_w = args.cell
    cell_h = int(args.cell * 9 / 16)
    label_h = 36
    cols = 3
    rows = (len(panels) + cols - 1) // cols

    pad = 12
    W = cols * cell_w + (cols + 1) * pad
    H = rows * (cell_h + label_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for idx, (title, path) in enumerate(panels):
        r, c = divmod(idx, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + label_h + pad)
        img = load(path, (cell_w, cell_h))
        canvas.paste(img, (x, y))
        ty = y + cell_h + 4
        draw.text((x + 6, ty), title, fill=(20, 20, 20), font=font)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out, optimize=True)
    print("wrote", args.out, canvas.size)


if __name__ == "__main__":
    main()
