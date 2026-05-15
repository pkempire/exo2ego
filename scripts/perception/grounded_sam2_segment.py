#!/usr/bin/env python3
"""Open-vocabulary detection + SAM2 box-prompt segmentation.

Pipeline:
  image
    -> GroundingDINO via Hugging Face Transformers zero-shot-object-detection
    -> SAM2 image predictor with each box as a prompt
    -> per-object masks, overlay, and JSON manifest

This is the "real model" replacement for the older VLM-box-only path. It is
still not perfect: open-vocabulary labels are brittle, and SAM2 inherits any
bad boxes from GroundingDINO. But the masks are model-produced pixels, not
hand-coded HSV or VLM rectangles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_LABELS = [
    "left hand",
    "right hand",
    "hand",
    "orange-spined book",
    "thin black book",
    "black upright book",
    "round wooden table",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    p.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam2-model", default="facebook/sam2-hiera-tiny")
    p.add_argument("--box-threshold", type=float, default=0.18)
    p.add_argument("--device", default="mps")
    return p.parse_args()


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_grounding_dino(image_path: Path, labels: list[str], model_name: str, device: str, threshold: float) -> list[dict]:
    from transformers import pipeline

    try:
        detector = pipeline("zero-shot-object-detection", model=model_name, device=device)
    except Exception:
        detector = pipeline("zero-shot-object-detection", model=model_name)

    pil = Image.open(image_path).convert("RGB")
    # HF expects labels ending in punctuation for GroundingDINO-style prompts.
    candidates = [label if label.endswith(".") else f"{label}." for label in labels]
    try:
        raw = detector(pil, candidate_labels=candidates, threshold=threshold)
    except NotImplementedError as exc:
        if device != "cpu" and ("MPS" in str(exc) or "mps" in str(exc)):
            print("GroundingDINO hit an MPS unsupported op; retrying detector on CPU.")
            detector = pipeline("zero-shot-object-detection", model=model_name, device="cpu")
            raw = detector(pil, candidate_labels=candidates, threshold=threshold)
        else:
            raise
    detections = []
    for item in raw:
        box = item.get("box") or {}
        label = str(item.get("label", "")).rstrip(".")
        score = float(item.get("score", 0.0))
        if not box:
            continue
        detections.append({
            "label": label,
            "score": score,
            "box_xyxy": [
                float(box["xmin"]),
                float(box["ymin"]),
                float(box["xmax"]),
                float(box["ymax"]),
            ],
        })
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def run_sam2(rgb: np.ndarray, detections: list[dict], model_id: str, device: str) -> list[dict]:
    import torch
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam_device = device if device in {"cuda", "cpu", "mps"} else "cpu"
    try:
        model = build_sam2_hf(model_id, device=sam_device)
    except TypeError:
        model = build_sam2_hf(model_id)
        model.to(sam_device)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(rgb)

    results = []
    for idx, det in enumerate(detections):
        box = np.array(det["box_xyxy"], dtype=np.float32)
        with torch.inference_mode():
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=True,
            )
        best = int(np.argmax(scores))
        mask = masks[best].astype(bool)
        ys, xs = np.where(mask)
        bbox = [0.0, 0.0, 0.0, 0.0]
        if len(xs):
            bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        results.append({
            **det,
            "sam2_score": float(scores[best]),
            "mask_area_px": int(mask.sum()),
            "mask_bbox_xyxy": bbox,
            "_mask": mask,
        })
    return results


def draw_overlay(rgb: np.ndarray, masks: list[dict]) -> np.ndarray:
    out = rgb.copy()
    colors = [
        (255, 80, 80),
        (80, 255, 120),
        (80, 180, 255),
        (255, 220, 80),
        (230, 90, 255),
        (80, 255, 240),
        (255, 150, 60),
    ]
    alpha = 0.42
    for idx, item in enumerate(masks):
        mask = item["_mask"]
        color = np.array(colors[idx % len(colors)], dtype=np.uint8)
        out[mask] = (out[mask].astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)
        x0, y0, x1, y1 = [int(v) for v in item["box_xyxy"]]
        cv2.rectangle(out, (x0, y0), (x1, y1), tuple(int(c) for c in color.tolist()), 3)
        label = f"{item['label']} {item['score']:.2f}/{item['sam2_score']:.2f}"
        cv2.putText(out, label, (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, tuple(int(c) for c in color.tolist()), 2)
    return out


def save_masks(out_dir: Path, masks: list[dict], rgb_shape: tuple[int, int]) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, item in enumerate(masks):
        mask = item.pop("_mask")
        safe = item["label"].replace(" ", "_").replace("/", "_")
        mask_path = out_dir / f"{idx:02d}_{safe}.png"
        cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
        rec = dict(item)
        rec["mask_path"] = str(mask_path)
        rec["mask_area_fraction"] = float(mask.mean())
        records.append(rec)
    return records


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = _area([x0, y0, x1, y1])
    union = _area(a) + _area(b) - inter
    return inter / max(union, 1e-6)


def canonical_role(label: str) -> str:
    l = label.lower()
    if "table" in l:
        return "table"
    if "orange" in l and "book" in l:
        return "orange_book"
    if "black" in l and "book" in l:
        return "black_book_or_device"
    if "hand" in l:
        return "hand"
    return l.replace(" ", "_")


def postprocess_segments(masks: list[dict], image_shape_hw: tuple[int, int]) -> list[dict]:
    """Remove duplicate/obviously bad detections and assign stable roles.

    GroundingDINO often returns overlapping labels ("hand", "left hand",
    "right hand") for the same physical hand. It can also return person-sized
    boxes for "hand". For this pipeline we need object-level masks, not every
    raw phrase match.
    """
    h, w = image_shape_hw
    image_area = float(h * w)
    cleaned = []
    for item in masks:
        role = canonical_role(item["label"])
        frac = float(item["mask_area_px"]) / image_area
        box = item["box_xyxy"]
        if role == "hand":
            box_frac = _area(item["mask_bbox_xyxy"]) / image_area
            # Ego-view hands can include forearm and exceed 4% of the image.
            # Reject only person/torso-sized false positives.
            if frac > 0.12 or (box_frac > 0.25 and frac > 0.04):
                continue
        if role in {"orange_book", "black_book_or_device"} and frac > 0.06:
            continue
        item["role"] = role
        cleaned.append(item)

    selected: list[dict] = []
    def rank_score(m: dict, role: str) -> float:
        base = m["score"] * 0.55 + m["sam2_score"] * 0.45
        x0, y0, x1, y1 = m["mask_bbox_xyxy"]
        bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        aspect = bh / bw
        if role == "orange_book":
            # The orange spine is a tall skinny strip; GroundingDINO also
            # returns the full book for this phrase, which is less useful for
            # preserving left/right object order.
            spine_bonus = min(aspect / 5.0, 1.0)
            small_bonus = 1.0 if m["mask_area_px"] / image_area < 0.012 else 0.0
            return base + 0.35 * spine_bonus + 0.25 * small_bonus
        return base

    for role in ["hand", "orange_book", "black_book_or_device", "table"]:
        candidates = [m for m in cleaned if m["role"] == role]
        candidates.sort(key=lambda m: rank_score(m, role), reverse=True)
        role_selected = []
        max_count = 2 if role == "hand" else 1
        for item in candidates:
            if any(_iou(item["mask_bbox_xyxy"], other["mask_bbox_xyxy"]) > 0.65 for other in role_selected):
                continue
            role_selected.append(item)
            if len(role_selected) >= max_count:
                break
        selected.extend(role_selected)

    # Stable order: hands left-to-right, then objects.
    def key(item: dict):
        role_order = {"hand": 0, "black_book_or_device": 1, "orange_book": 2, "table": 3}
        box = item["mask_bbox_xyxy"]
        return (role_order.get(item["role"], 9), box[0])

    return sorted(selected, key=key)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rgb = load_rgb(image_path)
    detections = run_grounding_dino(image_path, args.labels, args.dino_model, args.device, args.box_threshold)
    if not detections:
        raise SystemExit("GroundingDINO returned no detections; lower --box-threshold or adjust labels.")
    masks = run_sam2(rgb, detections, args.sam2_model, args.device)
    masks = postprocess_segments(masks, rgb.shape[:2])
    overlay = draw_overlay(rgb, masks)
    cv2.imwrite(str(out / "grounded_sam2_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    records = save_masks(out / "masks", masks, rgb.shape[:2])
    manifest = {
        "image": str(image_path),
        "labels": args.labels,
        "dino_model": args.dino_model,
        "sam2_model": args.sam2_model,
        "box_threshold": args.box_threshold,
        "device": args.device,
        "image_shape_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
        "segments": records,
    }
    (out / "segments.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out / 'segments.json'}")
    print(f"Wrote {out / 'grounded_sam2_overlay.png'}")


if __name__ == "__main__":
    main()
