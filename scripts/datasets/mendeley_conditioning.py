#!/usr/bin/env python3
"""Build interpretable conditioning for the Mendeley paired exo/ego dataset.

This uses deliberately simple CV cues so the report can explain every step:
skin-colored hand regions, bright cup/dice/object components, and the blue
table-cover region. The output is a scene graph, an overlay, and a prompt that
can be passed to an image model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def components(mask: np.ndarray, min_area: float = 80) -> list[dict]:
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-6:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        out.append(
            {
                "bbox_px": [int(x), int(y), int(w), int(h)],
                "centroid_px": [float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])],
                "area_px": area,
            }
        )
    return sorted(out, key=lambda item: item["area_px"], reverse=True)


def normalize_item(item: dict, image_shape: tuple[int, int]) -> dict:
    h, w = image_shape
    x, y, bw, bh = item["bbox_px"]
    cx, cy = item["centroid_px"]
    item = dict(item)
    item["bbox_norm"] = [round(x / w, 3), round(y / h, 3), round(bw / w, 3), round(bh / h, 3)]
    item["centroid_norm"] = [round(cx / w, 3), round(cy / h, 3)]
    item["area_frac"] = round(item["area_px"] / (w * h), 5)
    return item


def scene_graph(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    yy, xx = np.indices((h, w))

    # The table cover is pale cyan/blue. Use a broad color prior, then fill
    # the convex hull of the largest tabletop-like component.
    table_seed = (
        (yy > 0.18 * h)
        & (hsv[:, :, 2] > 45)
        & (g.astype(np.int16) > r.astype(np.int16) + 2)
        & (b.astype(np.int16) > r.astype(np.int16) - 8)
    )
    table_seed = cv2.morphologyEx(table_seed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    table_items = components(table_seed.astype(bool), min_area=2000)
    table_mask = np.zeros((h, w), dtype=np.uint8)
    if table_items:
        x, y, bw, bh = table_items[0]["bbox_px"]
        # Slightly expand the table hull; this gives later detectors a stable ROI.
        x0 = max(0, x - int(0.02 * w))
        y0 = max(0, y - int(0.02 * h))
        x1 = min(w, x + bw + int(0.02 * w))
        y1 = min(h, y + bh + int(0.02 * h))
        table_mask[y0:y1, x0:x1] = 1
    table_mask_bool = table_mask.astype(bool)
    table = components(table_mask_bool, min_area=1000)[:1]
    table_near = cv2.dilate(table_mask, np.ones((45, 45), np.uint8)).astype(bool)

    # Skin/hand regions. This is intentionally broad for low light.
    skin_mask = (
        (yy > 0.24 * h)
        &
        (hsv[:, :, 0] >= 0)
        & (hsv[:, :, 0] <= 28)
        & (hsv[:, :, 1] >= 25)
        & (hsv[:, :, 1] <= 190)
        & (hsv[:, :, 2] >= 35)
        & table_near
    )
    skin_mask = cv2.morphologyEx(skin_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    hands = []
    for item in components(skin_mask, min_area=600):
        area = item["area_px"]
        x, y, bw, bh = item["bbox_px"]
        if area > 0.08 * w * h:
            continue
        if bh < 16 or bw < 16:
            continue
        hands.append(item)
    hands = hands[:6]

    # White cups and dice candidates: bright, low-saturation components.
    bright_mask = (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 95)
    bright_mask &= table_near
    bright_mask &= ~((xx < 0.35 * w) & (yy < 0.12 * h))
    bright_mask = cv2.morphologyEx(bright_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    bright = components(bright_mask, min_area=30)
    cups = []
    dice = []
    for item in bright:
        x, y, bw, bh = item["bbox_px"]
        area = item["area_px"]
        aspect = bw / max(bh, 1)
        if area > 700 and 0.45 <= aspect <= 1.8:
            cups.append(item)
        elif 20 <= area <= 700 and 0.45 <= aspect <= 2.2:
            # Dice are small white square-ish bits, usually on table.
            if table_mask_bool[int(item["centroid_px"][1]), int(item["centroid_px"][0])]:
                dice.append(item)
    cups = cups[:4]
    dice = dice[:12]

    def norm_list(items: list[dict]) -> list[dict]:
        return [normalize_item(item, (h, w)) for item in items]

    return {
        "image": {"width": w, "height": h},
        "detectors": {
            "table_cover": "HSV cyan mask + morphology + largest component",
            "hands": "HSV skin mask + morphology + connected components",
            "cups": "low-saturation bright connected components, medium area",
            "dice": "low-saturation bright small components inside table mask",
        },
        "entities": {
            "table_cover": norm_list(table),
            "hand_regions": norm_list(hands),
            "cup_candidates": norm_list(cups),
            "dice_candidates": norm_list(dice),
        },
        "geometry_hints": {
            "source_view": "overhead/exocentric tabletop view",
            "target_view": "first-person ego view looking down at same table",
            "body_side": "bottom of ego frame",
            "far_side": "top of ego frame",
        },
    }


def draw_overlay(rgb: np.ndarray, sg: dict, out: Path) -> None:
    canvas = rgb.copy()
    colors = {
        "table_cover": (0, 220, 255),
        "hand_regions": (255, 190, 70),
        "cup_candidates": (255, 255, 255),
        "dice_candidates": (80, 255, 120),
    }
    for group, items in sg["entities"].items():
        color = colors[group]
        for idx, item in enumerate(items):
            x, y, w, h = item["bbox_px"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
            cv2.putText(canvas, f"{group}:{idx}", (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title("Conventional CV conditioning overlay")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def prompt_from_scene(sg: dict) -> str:
    entities = sg["entities"]
    lines = [
        "Transform the attached exocentric tabletop image into a matching first-person egocentric camera view.",
        "",
        "Use these measured scene cues from simple computer vision:",
        f"- image_size: {sg['image']['width']}x{sg['image']['height']}",
        f"- table_cover regions: {[x['bbox_norm'] for x in entities['table_cover']]}",
        f"- hand/forearm regions: {[x['bbox_norm'] for x in entities['hand_regions'][:4]]}",
        f"- cup candidates: {[x['bbox_norm'] for x in entities['cup_candidates'][:3]]}",
        f"- dice/small white object candidates: {[x['bbox_norm'] for x in entities['dice_candidates'][:8]]}",
        "",
        "Target camera geometry:",
        "- Output should look like a real first-person phone/action-camera frame looking downward at the table.",
        "- The body-side table edge should be at the bottom of the image.",
        "- The far side of the table and background should be toward the top.",
        "- Hands and forearms should enter from the lower or side edges.",
        "- Preserve the blue Toy Story table cover, white cup(s), dice, hands/forearms, dim indoor lighting, and object layout.",
        "- Do not add faces, identifying features, extra people, or a clean studio scene.",
        "- Keep the low-light phone-camera feel rather than making it a perfect product render.",
        "",
        "Output one natural photorealistic first-person tabletop image.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(args.image)
    sg = scene_graph(rgb)
    (out / "scene_graph.json").write_text(json.dumps(sg, indent=2))
    draw_overlay(rgb, sg, out / "conditioning_overlay.png")
    (out / "prompt_conditioned.txt").write_text(prompt_from_scene(sg) + "\n")
    print(f"Wrote {out / 'scene_graph.json'}")
    print(f"Wrote {out / 'conditioning_overlay.png'}")
    print(f"Wrote {out / 'prompt_conditioned.txt'}")


if __name__ == "__main__":
    main()
