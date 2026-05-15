#!/usr/bin/env python3
"""Build a dense mask-aware ego condition image from exo segments.

This is not a solved geometric reconstruction. It is a stronger conditioning
artifact than the sparse point cloud: it uses real GroundingDINO+SAM2 masks to
crop the manipulation-relevant source pixels and place them into a canonical
first-person layout while preserving relative contact/order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--segments", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=1024)
    return p.parse_args()


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: str | Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def alpha_paste(canvas: np.ndarray, crop: np.ndarray, mask: np.ndarray, center: tuple[int, int], max_size: tuple[int, int]) -> None:
    ys, xs = np.where(mask)
    if not len(xs):
        return
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    crop = crop[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    h, w = crop.shape[:2]
    scale = min(max_size[0] / max(w, 1), max_size[1] / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST) > 127
    cx, cy = center
    tx0, ty0 = cx - nw // 2, cy - nh // 2
    tx1, ty1 = tx0 + nw, ty0 + nh
    sx0, sy0 = max(0, -tx0), max(0, -ty0)
    sx1, sy1 = nw - max(0, tx1 - canvas.shape[1]), nh - max(0, ty1 - canvas.shape[0])
    tx0, ty0 = max(0, tx0), max(0, ty0)
    tx1, ty1 = tx0 + (sx1 - sx0), ty0 + (sy1 - sy0)
    if tx1 <= tx0 or ty1 <= ty0:
        return
    m = mask[sy0:sy1, sx0:sx1]
    canvas[ty0:ty1, tx0:tx1][m] = crop[sy0:sy1, sx0:sx1][m]


def choose_segments(segments: list[dict]) -> dict[str, list[dict]]:
    roles: dict[str, list[dict]] = {}
    for s in segments:
        roles.setdefault(s.get("role", s.get("label", "unknown")), []).append(s)
    for role in roles:
        roles[role].sort(key=lambda s: (s.get("score", 0) * 0.5 + s.get("sam2_score", 0) * 0.5), reverse=True)
    return roles


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(args.image)
    data = json.loads(Path(args.segments).read_text())
    roles = choose_segments(data["segments"])
    S = args.size

    # Warm table-color background from the table mask when available.
    canvas = np.zeros((S, S, 3), dtype=np.uint8)
    table_color = np.array([120, 85, 55], dtype=np.uint8)
    if roles.get("table"):
        m = load_mask(roles["table"][0]["mask_path"])
        pix = rgb[m]
        if len(pix):
            table_color = np.median(pix, axis=0).astype(np.uint8)
    canvas[:] = np.array([28, 28, 30], dtype=np.uint8)
    cv2.ellipse(canvas, (S // 2, int(S * 0.62)), (int(S * 0.58), int(S * 0.34)), 0, 0, 360, tuple(int(x) for x in table_color.tolist()), -1)

    # Paste objects first, hands second so contact hands stay visible.
    object_specs = [
        ("black_book_or_device", (int(S * 0.58), int(S * 0.42)), (int(S * 0.18), int(S * 0.50))),
        ("orange_book", (int(S * 0.68), int(S * 0.42)), (int(S * 0.18), int(S * 0.50))),
    ]
    pasted = []
    for role, center, max_size in object_specs:
        if roles.get(role):
            seg = roles[role][0]
            alpha_paste(canvas, rgb, load_mask(seg["mask_path"]), center, max_size)
            pasted.append(role)

    hands = roles.get("hand", [])
    # left-to-right in source; place source-left hand lower-left, source-right/contact hand near objects.
    hands = sorted(hands, key=lambda s: s["mask_bbox_xyxy"][0])
    hand_targets = [
        ((int(S * 0.30), int(S * 0.78)), (int(S * 0.30), int(S * 0.28))),
        ((int(S * 0.56), int(S * 0.55)), (int(S * 0.26), int(S * 0.35))),
    ]
    for seg, (center, max_size) in zip(hands, hand_targets):
        alpha_paste(canvas, rgb, load_mask(seg["mask_path"]), center, max_size)

    condition_path = out / "mask_ego_condition.png"
    cv2.imwrite(str(condition_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    prompt = (
        "This image is a mask-composed egocentric condition built from real SAM2 segments of an exocentric frame. "
        "Turn it into a photorealistic first-person head-mounted view. Preserve the physical constraints: "
        "two hands if visible, a round wooden table, an upright black book/device next to an orange-spined book, "
        "and right-hand contact with the upright black object. Do not flip the side of the table; keep the viewer "
        "seated on the same side as the hands, looking down at the object interaction."
    )
    (out / "mask_condition_prompt.txt").write_text(prompt + "\n")
    (out / "mask_condition_meta.json").write_text(json.dumps({
        "image": args.image,
        "segments": args.segments,
        "condition": str(condition_path),
        "roles_available": {k: len(v) for k, v in roles.items()},
        "pasted_roles": pasted + ["hand"] * min(len(hands), 2),
    }, indent=2))
    print(f"Wrote {condition_path}")
    print(f"Wrote {out / 'mask_condition_prompt.txt'}")


if __name__ == "__main__":
    main()
