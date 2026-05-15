#!/usr/bin/env python3
"""Generic VLM scene-graph extraction for exo-to-ego prompting.

This is the model-based replacement for hardcoded HSV scripts. It asks a
vision model to produce normalized boxes and manipulation-relevant facts:
hands, objects, table/workspace, contact, relative layout, and target POV
constraints. It then renders an overlay and writes a prompt for image
generation.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gpt-5.1")
    return parser.parse_args()


def data_url(path: str | Path) -> str:
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def strip_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def prompt() -> str:
    return """
You are extracting a scene graph for an exocentric-to-egocentric view synthesis system.

Return strict JSON only, with this schema:
{
  "image_summary": "short description",
  "camera": {
    "source_view": "exo/overhead/front/etc",
    "likely_target_ego_view": "what the first-person camera should see",
    "body_side_in_target": "bottom/left/right/unknown",
    "far_side_in_target": "top/left/right/unknown"
  },
  "workspace": {
    "name": "table/counter/etc",
    "bbox_norm": [x, y, w, h],
    "appearance": "short visual description"
  },
  "hands": [
    {
      "name": "left/right/unknown hand",
      "bbox_norm": [x, y, w, h],
      "state": "grasping/resting/reaching/etc",
      "near_object": "object name or null"
    }
  ],
  "objects": [
    {
      "name": "specific object name",
      "bbox_norm": [x, y, w, h],
      "appearance": "color/shape/material",
      "role": "manipulated/support/background",
      "relative_position": "left/right/center/near/far relationship",
      "contact_with_hand": true/false
    }
  ],
  "spatial_constraints": [
    "plain-language constraints useful for generation"
  ],
  "failure_risks": [
    "what an image generator is likely to get wrong"
  ]
}

Rules:
- bbox_norm values must be normalized to [0,1] as [x_min, y_min, width, height].
- Include only visible objects relevant to manipulation and viewpoint reconstruction.
- Do not identify people. Use hands/forearms only.
- Be conservative if uncertain.
""".strip()


def build_generation_prompt(scene: dict) -> str:
    lines = [
        "Transform the attached exocentric image into a realistic first-person egocentric camera view.",
        "",
        "Use this extracted scene graph:",
        json.dumps(scene, indent=2),
        "",
        "Generation requirements:",
        "- Preserve all manipulation-relevant objects and hands/forearms.",
        "- Preserve left/right and near/far layout relationships from the scene graph.",
        "- Put the body-side workspace near the bottom of the ego image when indicated.",
        "- Keep lighting, camera quality, and environment consistent with the source.",
        "- Do not add faces, identifying features, extra hands, or unrelated objects.",
        "- If unseen regions are required, hallucinate plausibly but do not move the main objects.",
        "",
        "Output one natural photorealistic first-person image.",
    ]
    return "\n".join(lines)


def draw_overlay(image_path: Path, scene: dict, out_path: Path) -> None:
    rgb = load_rgb(image_path)
    h, w = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(rgb)
    ax.axis("off")

    def draw_box(box, label, color):
        if not box or len(box) != 4:
            return
        x, y, bw, bh = box
        rect = plt.Rectangle((x * w, y * h), bw * w, bh * h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x * w, max(12, y * h - 4), label, color=color, fontsize=8, bbox={"facecolor": "black", "alpha": 0.35, "pad": 1})

    workspace = scene.get("workspace") or {}
    draw_box(workspace.get("bbox_norm"), workspace.get("name", "workspace"), "cyan")
    for hand in scene.get("hands", []):
        draw_box(hand.get("bbox_norm"), hand.get("name", "hand"), "yellow")
    for obj in scene.get("objects", []):
        color = "lime" if obj.get("role") == "manipulated" else "white"
        draw_box(obj.get("bbox_norm"), obj.get("name", "object"), color)

    ax.set_title("VLM scene graph overlay")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Put it in .env first.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_path = Path(args.image)
    client = OpenAI()
    response = client.responses.create(
        model=args.model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt()},
                    {"type": "input_image", "image_url": data_url(image_path)},
                ],
            }
        ],
    )
    scene = strip_json(response.output_text)
    (out / "vlm_scene_graph.json").write_text(json.dumps(scene, indent=2))
    (out / "prompt_conditioned_vlm.txt").write_text(build_generation_prompt(scene) + "\n")
    draw_overlay(image_path, scene, out / "vlm_scene_graph_overlay.png")
    print(f"Wrote {out / 'vlm_scene_graph.json'}")
    print(f"Wrote {out / 'prompt_conditioned_vlm.txt'}")
    print(f"Wrote {out / 'vlm_scene_graph_overlay.png'}")


if __name__ == "__main__":
    main()
