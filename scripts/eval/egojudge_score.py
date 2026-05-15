#!/usr/bin/env python3
"""Score ego-view candidates with simple geometry/layout checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def entities(mask: np.ndarray) -> list[dict]:
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 50:
            continue
        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-6:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        items.append({"centroid": [float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])], "bbox": [x, y, w, h], "area": area})
    return sorted(items, key=lambda item: item["area"], reverse=True)


def entity(mask: np.ndarray) -> dict | None:
    items = entities(mask)
    return items[0] if items else None


def dark_object(mask: np.ndarray, shape: tuple[int, int], orange: dict | None = None) -> dict | None:
    h, w = shape
    orange_x = orange["centroid"][0] if orange else w * 0.4
    best = None
    best_score = -1.0
    for item in entities(mask):
        x, y, bw, bh = item["bbox"]
        cx, cy = item["centroid"]
        area_frac = item["area"] / (w * h)
        aspect = bw / max(bh, 1)
        if not (0.001 <= area_frac <= 0.08):
            continue
        if not (0.25 <= aspect <= 4.0):
            continue
        # Favor the mug-like dark object on the right of the manipulated carton.
        score = item["area"] + 0.5 * max(0, cx - orange_x)
        if cx > orange_x and score > best_score:
            best = item
            best_score = score
    return best


def features(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yy = np.indices((h, w))[0]
    orange_mask = (hsv[:, :, 0] >= 3) & (hsv[:, :, 0] <= 30) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 80)
    dark_mask = (hsv[:, :, 2] < 175) & (hsv[:, :, 1] < 125) & (yy > h * 0.15)
    skin_mask = (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] >= 30) & (hsv[:, :, 1] <= 190) & (hsv[:, :, 2] >= 80)

    orange = entity(orange_mask)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    feats = {
        "size": [w, h],
        "orange": orange,
        "dark": dark_object(dark_mask, (h, w), orange),
        "skin": entity(skin_mask),
        "nonblack_frac": float((gray > 12).mean()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }
    for item in ["orange", "dark", "skin"]:
        if feats[item]:
            cx, cy = feats[item]["centroid"]
            feats[item]["centroid_norm"] = [cx / w, cy / h]
            feats[item]["area_frac"] = feats[item]["area"] / (w * h)
    return feats


def score_candidate(source: dict, candidate: dict) -> dict:
    score = 0.0
    reasons = []
    w, h = candidate["size"]
    aspect = w / max(h, 1)

    if candidate["nonblack_frac"] > 0.92:
        score += 10
        reasons.append("full image coverage")

    if aspect > 2.25:
        score -= 35
        reasons.append("likely comparison/debug panel rather than a single ego image")

    if candidate["orange"]:
        score += 25
        reasons.append("orange manipulated object detected")
    else:
        reasons.append("missing orange manipulated object")

    if candidate["dark"]:
        score += 15
        reasons.append("dark mug/object detected")
    else:
        reasons.append("missing dark mug/object")

    if candidate["skin"]:
        score += 15
        reasons.append("hands/skin detected")
    else:
        reasons.append("hands/skin weak or missing")

    if candidate["orange"] and candidate["dark"]:
        if candidate["orange"]["centroid_norm"][0] < candidate["dark"]["centroid_norm"][0]:
            score += 20
            reasons.append("layout preserved: orange object left of mug/object")
        else:
            reasons.append("layout mismatch: mug/object not right of orange object")

    if candidate["orange"] and candidate["skin"]:
        ox, oy = candidate["orange"]["centroid_norm"]
        sx, sy = candidate["skin"]["centroid_norm"]
        dist = float(np.hypot(ox - sx, oy - sy))
        if dist < 0.35:
            score += 15
            reasons.append("hand/object proximity plausible")
        else:
            reasons.append("hand/object proximity weak")

    sharpness = candidate.get("sharpness", 0.0)
    if sharpness > 80:
        score += 10
        reasons.append("reasonable image sharpness")
    elif sharpness < 25:
        score -= 10
        reasons.append("very blurry candidate")

    score = float(np.clip(score, 0, 100))
    return {"score": round(score, 2), "reasons": reasons, "features": candidate}


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    source_features = features(load_rgb(args.source))
    results = {"source": str(Path(args.source).resolve()), "source_features": source_features, "candidates": {}}

    for cand in args.candidates:
        path = Path(cand)
        if not path.exists():
            continue
        cand_features = features(load_rgb(str(path)))
        results["candidates"][path.name] = score_candidate(source_features, cand_features)

    (out / "scores.json").write_text(json.dumps(results, indent=2))

    lines = ["# EgoJudge Scores", ""]
    for name, item in sorted(results["candidates"].items(), key=lambda kv: kv[1]["score"], reverse=True):
        lines.append(f"## {name}")
        lines.append(f"Score: {item['score']}/100")
        for reason in item["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines))
    print(f"Wrote {out / 'scores.json'}")
    print(f"Wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
