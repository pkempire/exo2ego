"""Open-vocab object detection over an exo frame using HuggingFace Inference API.

Replaces the GPT-4o "scene graph" bbox hallucinations with real Grounding-DINO boxes.

Usage:
    python scripts/detect_hosted_hf.py \
        --image data/own_capture/take1/exo_frames/exo_013.jpg \
        --out   data/own_capture/take1/hf_detect_exo_013 \
        --labels "hand. book. table. couch. window. tv. person."

The script writes:
    detections.json      raw boxes + scores + labels
    scene_graph.json     same schema as scripts/vlm_scene_graph.py output (normalized bboxes)
    overlay.png          source image with boxes drawn on top
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv


def hf_grounding_dino(
    image_path: str,
    labels: str,
    token: str,
    model: str = "IDEA-Research/grounding-dino-tiny",
    threshold: float = 0.25,
) -> dict:
    """Run Grounding-DINO via the transformers library (downloads weights via HF token).

    The HF Inference Providers route does not serve grounding-dino (returns
    "Model not supported by provider hf-inference"). The model weights are
    ~170MB and run fine on CPU, so we pull them once and infer locally.
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import torch

    os.environ["HF_TOKEN"] = token  # for hub auth on first download
    processor = AutoProcessor.from_pretrained(model)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(model)
    detector.eval()

    img = Image.open(image_path).convert("RGB")
    # Grounding-DINO expects labels as a single string ending in periods.
    text = labels if labels.strip().endswith(".") else labels.strip() + "."

    inputs = processor(images=img, text=text, return_tensors="pt")
    with torch.no_grad():
        outputs = detector(**inputs)

    target_sizes = torch.tensor([img.size[::-1]])  # (H, W)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=threshold,
        text_threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        detections.append({
            "label": str(label),
            "score": float(score),
            "box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
        })
    return {"endpoint": f"local:{model}", "result": detections}


def to_scene_graph(detections: list, W: int, H: int) -> dict:
    """Convert HF detection list to the scene_graph.json shape vlm_scene_graph.py emits."""
    hands, objects, workspace = [], [], None
    for det in detections:
        label = det.get("label", "").lower().strip(". ")
        score = float(det.get("score", 0.0))
        b = det.get("box") or det.get("bbox") or {}
        if not b:
            continue
        x1 = float(b.get("xmin", b.get("x1", 0))) / W
        y1 = float(b.get("ymin", b.get("y1", 0))) / H
        x2 = float(b.get("xmax", b.get("x2", 0))) / W
        y2 = float(b.get("ymax", b.get("y2", 0))) / H
        bbox_norm = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
        entry = {
            "name": label,
            "bbox_norm": bbox_norm,
            "score": round(score, 3),
        }
        if "hand" in label:
            hands.append({**entry, "state": "unknown", "near_object": None})
        elif "table" in label or "desk" in label or "counter" in label:
            if workspace is None or score > workspace.get("_score", 0):
                workspace = {**entry, "_score": score, "appearance": "detected"}
        else:
            objects.append({**entry, "role": "background"})
    return {
        "image_summary": "auto-detected by Grounding-DINO",
        "camera": {
            "source_view": "exo",
            "likely_target_ego_view": "wearer view of workspace",
            "body_side_in_target": "bottom",
            "far_side_in_target": "top",
        },
        "workspace": workspace or {"name": "unknown", "bbox_norm": [0, 0, 1, 1]},
        "hands": hands,
        "objects": objects,
        "spatial_constraints": [],
        "failure_risks": [],
    }


def draw_overlay(image_path: str, detections: list, out_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    palette = [(255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 165, 0), (200, 0, 200), (0, 200, 200)]
    for i, det in enumerate(detections):
        b = det.get("box") or det.get("bbox") or {}
        if not b:
            continue
        x1, y1 = float(b.get("xmin", b.get("x1", 0))), float(b.get("ymin", b.get("y1", 0)))
        x2, y2 = float(b.get("xmax", b.get("x2", 0))), float(b.get("ymax", b.get("y2", 0)))
        color = palette[i % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f"{det.get('label','')}  {det.get('score',0):.2f}"
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        draw.rectangle([x1, y1, x1 + tw + 8, y1 + th + 4], fill=color)
        draw.text((x1 + 4, y1 + 2), label, fill=(255, 255, 255), font=font)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default="hand. book. table. window. tv. couch. person.")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--threshold", type=float, default=0.25)
    args = ap.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN in .env", file=sys.stderr)
        sys.exit(2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = hf_grounding_dino(args.image, args.labels, token, args.model, args.threshold)
    (out / "raw_response.json").write_text(json.dumps(raw, indent=2))

    detections = raw["result"]
    if isinstance(detections, dict) and "error" in detections:
        print("HF error:", detections["error"], file=sys.stderr)
        sys.exit(3)
    if not isinstance(detections, list):
        detections = detections.get("predictions", [])

    (out / "detections.json").write_text(json.dumps(detections, indent=2))

    img = Image.open(args.image)
    W, H = img.size
    sg = to_scene_graph(detections, W, H)
    (out / "scene_graph.json").write_text(json.dumps(sg, indent=2))

    draw_overlay(args.image, detections, str(out / "overlay.png"))

    print(f"wrote {out}/ ({len(detections)} detections)")


if __name__ == "__main__":
    main()
