#!/usr/bin/env python3
"""Depth/physics-aware EgoJudge scorer.

This script uses a monocular depth model plus simple object/skin detectors to
check whether generated ego-view candidates are physically plausible enough to
be useful as robot-training data.

It is intentionally conservative: it does not prove a generation is correct,
but it catches common failures like missing objects, broken hand-object contact,
bad foreground/background ordering, and debug-panel artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    return parser.parse_args()


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def depth_pipe(model_name: str):
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = -1
    return pipeline("depth-estimation", model=model_name, device=device)


def predict_depth(pipe, rgb: np.ndarray) -> np.ndarray:
    result = pipe(Image.fromarray(rgb))
    pred = result.get("predicted_depth", result.get("depth"))
    if hasattr(pred, "detach"):
        depth = pred.detach().cpu().numpy()
    else:
        depth = np.array(pred)
    depth = np.squeeze(depth).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    finite = np.isfinite(depth)
    if finite.any():
        lo, hi = np.percentile(depth[finite], [2, 98])
        depth = np.clip(depth, lo, hi)
        depth = (depth - lo) / max(hi - lo, 1e-6)
    return depth


def components(mask: np.ndarray, min_area: float = 80) -> list[dict]:
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                "area": area,
                "bbox": [int(x), int(y), int(w), int(h)],
                "centroid": [float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])],
                "contour": contour,
            }
        )
    return sorted(out, key=lambda item: item["area"], reverse=True)


def detect_entities(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yy, xx = np.indices((h, w))

    orange_mask = (
        (hsv[:, :, 0] >= 2)
        & (hsv[:, :, 0] <= 32)
        & (hsv[:, :, 1] > 65)
        & (hsv[:, :, 2] > 75)
    )
    skin_mask = (
        (hsv[:, :, 0] <= 25)
        & (hsv[:, :, 1] >= 28)
        & (hsv[:, :, 1] <= 205)
        & (hsv[:, :, 2] >= 75)
    )
    dark_mask = (
        (hsv[:, :, 2] < 180)
        & (hsv[:, :, 1] < 135)
        & (yy > h * 0.10)
    )

    orange = components(orange_mask, min_area=0.0005 * w * h)
    orange_obj = orange[0] if orange else None

    dark_obj = None
    best_score = -1.0
    orange_x = orange_obj["centroid"][0] if orange_obj else w * 0.45
    for item in components(dark_mask, min_area=0.0005 * w * h):
        x, y, bw, bh = item["bbox"]
        cx, cy = item["centroid"]
        area_frac = item["area"] / (w * h)
        aspect = bw / max(bh, 1)
        if not (0.001 <= area_frac <= 0.12):
            continue
        if not (0.20 <= aspect <= 5.0):
            continue
        score = item["area"] + 0.3 * max(0.0, cx - orange_x)
        if cx > orange_x and score > best_score:
            dark_obj = item
            best_score = score

    skin_components = components(skin_mask, min_area=0.0004 * w * h)
    # Keep skin components near the lower half / workspace. This avoids picking
    # random skin-colored background patches in generative outputs.
    skin_components = [c for c in skin_components if c["centroid"][1] > h * 0.35]
    skin_obj = skin_components[0] if skin_components else None

    for obj in [orange_obj, dark_obj, skin_obj]:
        if obj:
            cx, cy = obj["centroid"]
            obj["centroid_norm"] = [cx / w, cy / h]
            obj["area_frac"] = obj["area"] / (w * h)
            del obj["contour"]

    return {"orange": orange_obj, "mug_dark": dark_obj, "hands_skin": skin_obj}


def mean_depth(depth: np.ndarray, obj: dict | None) -> float | None:
    if not obj:
        return None
    x, y, w, h = obj["bbox"]
    patch = depth[y:y + h, x:x + w]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def score_depth_physics(rgb: np.ndarray, depth: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    entities = detect_entities(rgb)
    depths = {k: mean_depth(depth, v) for k, v in entities.items()}

    score = 0.0
    reasons = []

    aspect = w / max(h, 1)
    if aspect > 2.25:
        score -= 25
        reasons.append("candidate looks like a multi-panel/debug image")
    else:
        score += 8
        reasons.append("single-image aspect ratio plausible")

    if (gray > 12).mean() > 0.92:
        score += 8
        reasons.append("full-frame visual coverage")

    for key, label, pts in [
        ("orange", "manipulated orange object", 18),
        ("mug_dark", "dark mug/object", 12),
        ("hands_skin", "hands/forearms", 18),
    ]:
        if entities[key]:
            score += pts
            reasons.append(f"{label} detected")
        else:
            reasons.append(f"{label} missing/weak")

    orange = entities["orange"]
    mug = entities["mug_dark"]
    hands = entities["hands_skin"]

    if orange and mug:
        if orange["centroid_norm"][0] < mug["centroid_norm"][0]:
            score += 12
            reasons.append("left/right layout preserved: orange object left of mug")
        else:
            score -= 8
            reasons.append("layout violation: mug not right of orange object")

    if orange and hands:
        ox, oy = orange["centroid_norm"]
        hx, hy = hands["centroid_norm"]
        dist = float(np.hypot(ox - hx, oy - hy))
        if dist < 0.33:
            score += 14
            reasons.append("hand-object contact/proximity plausible")
        else:
            score -= 8
            reasons.append("hand-object contact/proximity implausible")

    # Depth Anything style outputs are relative; for our current model larger
    # values usually indicate closer image regions after normalization. We only
    # use coarse ordering, not metric distances.
    if depths["orange"] is not None and depths["hands_skin"] is not None:
        if abs(depths["orange"] - depths["hands_skin"]) < 0.28:
            score += 10
            reasons.append("depth: hands and manipulated object lie on similar foreground layer")
        else:
            reasons.append("depth: hands/object layer mismatch")

    if depths["orange"] is not None and depths["mug_dark"] is not None:
        if depths["orange"] >= depths["mug_dark"] - 0.18:
            score += 8
            reasons.append("depth: manipulated object is not behind the mug")
        else:
            score -= 5
            reasons.append("depth: manipulated object appears behind mug/object")

    if hands:
        # Ego view should usually put forearms/hands in the lower half.
        if hands["centroid_norm"][1] > 0.48:
            score += 8
            reasons.append("ego composition: hands appear in lower/workspace region")
        else:
            reasons.append("ego composition: hands too high for POV")

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness > 70:
        score += 5
        reasons.append("image has adequate sharpness")
    elif sharpness < 20:
        score -= 8
        reasons.append("image is very blurry")

    return {
        "score": round(float(np.clip(score, 0, 100)), 2),
        "reasons": reasons,
        "entities": entities,
        "relative_depths": depths,
        "sharpness": sharpness,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = depth_pipe(args.model)
    results = {"depth_model": args.model, "candidates": {}}

    for candidate in args.candidates:
        path = Path(candidate)
        if not path.exists():
            continue
        rgb = load_rgb(str(path))
        depth = predict_depth(pipe, rgb)
        item = score_depth_physics(rgb, depth)
        results["candidates"][path.name] = item

        depth_vis = (255 * depth).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(out / f"{path.stem}_depth.png"), depth_vis)

    (out / "depth_scores.json").write_text(json.dumps(results, indent=2))
    lines = ["# EgoJudge Depth/Physics Scores", ""]
    for name, item in sorted(results["candidates"].items(), key=lambda kv: kv[1]["score"], reverse=True):
        lines.append(f"## {name}")
        lines.append(f"Score: {item['score']}/100")
        for reason in item["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
    (out / "depth_report.md").write_text("\n".join(lines))
    print(f"Wrote {out / 'depth_scores.json'}")
    print(f"Wrote {out / 'depth_report.md'}")


if __name__ == "__main__":
    main()
