#!/usr/bin/env python3
"""Refine VLM bounding boxes into approximate masks using GrabCut.

This is not as good as SAM2, but it is a meaningful upgrade over broad boxes:
VLM gives object/hand boxes; GrabCut uses image edges/color statistics inside
each box to estimate a foreground mask and draw contours.
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
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-background", action="store_true")
    return parser.parse_args()


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def box_to_rect(box: list[float], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    if not box or len(box) != 4:
        return None
    x, y, bw, bh = box
    x0 = int(np.clip(x * w, 0, w - 1))
    y0 = int(np.clip(y * h, 0, h - 1))
    x1 = int(np.clip((x + bw) * w, x0 + 2, w))
    y1 = int(np.clip((y + bh) * h, y0 + 2, h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1 - x0, y1 - y0


def grabcut_mask(rgb: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        x, y, w, h = rect
        fallback = np.zeros(rgb.shape[:2], dtype=np.uint8)
        fallback[y : y + h, x : x + w] = 1
        return fallback.astype(bool)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


def entities(scene: dict, include_background: bool) -> list[tuple[str, str, list[float], str | None]]:
    out = []
    workspace = scene.get("workspace") or {}
    if workspace.get("bbox_norm"):
        out.append(("workspace", workspace.get("name", "workspace"), workspace["bbox_norm"], None))
    for item in scene.get("hands", []):
        out.append(("hand", item.get("name", "hand"), item.get("bbox_norm"), None))
    for item in scene.get("objects", []):
        role = item.get("role")
        if role == "background" and not include_background:
            continue
        out.append(("object", item.get("name", "object"), item.get("bbox_norm"), role))
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(args.image)
    scene = json.loads(Path(args.scene).read_text())
    h, w = rgb.shape[:2]

    overlay = rgb.copy().astype(np.float32)
    masks_json = []
    colors = {
        "workspace": np.array([0, 210, 255], dtype=np.float32),
        "hand": np.array([255, 210, 40], dtype=np.float32),
        "object": np.array([80, 255, 120], dtype=np.float32),
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(rgb)
    ax.axis("off")
    for kind, name, box, role in entities(scene, args.include_background):
        rect = box_to_rect(box, (h, w))
        if rect is None:
            continue
        if kind == "workspace":
            x, y, rw, rh = rect
            mask = np.zeros((h, w), dtype=bool)
            mask[y : y + rh, x : x + rw] = True
            rect_patch = plt.Rectangle((x, y), rw, rh, linewidth=2.2, edgecolor=colors[kind] / 255.0, facecolor="none")
            ax.add_patch(rect_patch)
            ax.text(x, max(12, y - 4), f"{kind}: {name}", color=colors[kind] / 255.0, fontsize=8, bbox={"facecolor": "black", "alpha": 0.35, "pad": 1})
            masks_json.append(
                {
                    "kind": kind,
                    "name": name,
                    "bbox_px": [int(x), int(y), int(rw), int(rh)],
                    "mask_area_px": int(mask.sum()),
                    "mask_area_frac": round(float(mask.mean()), 5),
                    "centroid_px": [float(x + rw / 2), float(y + rh / 2)],
                    "note": "workspace is drawn as an outline, not used as an object mask",
                }
            )
            continue
        else:
            mask = grabcut_mask(rgb, rect)
        color = colors[kind]
        overlay[mask] = 0.58 * overlay[mask] + 0.42 * color
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 30:
                continue
            contour = contour.squeeze(axis=1)
            ax.plot(contour[:, 0], contour[:, 1], color=color / 255.0, linewidth=1.7)
        x, y, rw, rh = rect
        ax.text(x, max(12, y - 4), f"{kind}: {name}", color=color / 255.0, fontsize=8, bbox={"facecolor": "black", "alpha": 0.35, "pad": 1})
        ys, xs = np.where(mask)
        masks_json.append(
            {
                "kind": kind,
                "name": name,
                "role": role,
                "bbox_px": [int(x), int(y), int(rw), int(rh)],
                "mask_area_px": int(mask.sum()),
                "mask_area_frac": round(float(mask.mean()), 5),
                "centroid_px": [float(xs.mean()), float(ys.mean())] if len(xs) else None,
            }
        )

    ax.imshow(np.clip(overlay / 255.0, 0, 1), alpha=0.45)
    ax.set_title("VLM boxes + GrabCut-refined mask overlay")
    fig.tight_layout()
    fig.savefig(out / "refined_mask_overlay.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    (out / "refined_masks.json").write_text(json.dumps({"masks": masks_json}, indent=2))
    print(f"Wrote {out / 'refined_mask_overlay.png'}")
    print(f"Wrote {out / 'refined_masks.json'}")


if __name__ == "__main__":
    main()
