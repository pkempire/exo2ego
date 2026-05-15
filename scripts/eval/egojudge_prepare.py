#!/usr/bin/env python3
"""Prepare EgoJudge prompts and experiment manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--phase2", default=None, help="Optional Phase 2 JSON with hands/depth.")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def component_candidates(mask: np.ndarray) -> list[dict]:
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-6:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        items.append(
            {
                "centroid_px": [float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])],
                "bbox_px": [int(x), int(y), int(w), int(h)],
                "area_px": area,
            }
        )
    return sorted(items, key=lambda item: item["area_px"], reverse=True)


def largest_component(mask: np.ndarray) -> dict | None:
    items = component_candidates(mask)
    if not items:
        return None
    return items[0]


def select_mug_candidate(mask: np.ndarray, image_shape: tuple[int, int]) -> dict | None:
    h, w = image_shape
    best = None
    best_score = -1.0
    for item in component_candidates(mask):
        x, y, bw, bh = item["bbox_px"]
        cx, cy = item["centroid_px"]
        area_frac = item["area_px"] / (w * h)
        aspect = bw / max(bh, 1)
        if not (0.001 <= area_frac <= 0.04):
            continue
        if not (0.35 <= aspect <= 3.5):
            continue
        if cx < 0.45 * w or cy < 0.50 * h:
            continue
        score = item["area_px"] + 0.25 * cx
        if score > best_score:
            best = item
            best_score = score
    return best


def detect_scene(rgb: np.ndarray, phase2_path: str | None) -> dict:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    orange = (
        (hsv[:, :, 0] >= 3)
        & (hsv[:, :, 0] <= 28)
        & (hsv[:, :, 1] > 80)
        & (hsv[:, :, 2] > 90)
    )
    orange_obj = largest_component(orange)

    yy, xx = np.indices((h, w))
    dark = (
        (hsv[:, :, 2] < 175)
        & (hsv[:, :, 1] < 115)
        & (yy > h * 0.50)
        & (xx > w * 0.45)
    )
    dark_obj = select_mug_candidate(dark, (h, w))

    skin = (
        (hsv[:, :, 0] >= 0)
        & (hsv[:, :, 0] <= 25)
        & (hsv[:, :, 1] >= 35)
        & (hsv[:, :, 1] <= 190)
        & (hsv[:, :, 2] >= 90)
    )
    skin_obj = largest_component(skin)

    hands = []
    phase2 = None
    depth_summary = None
    if phase2_path:
        p = Path(phase2_path)
        if p.exists():
            phase2 = json.loads(p.read_text())
            if "depth_map" in phase2:
                depth = np.array(phase2["depth_map"], dtype=np.float32)
                finite = np.isfinite(depth)
                depth_summary = {
                    "min": float(depth[finite].min()),
                    "median": float(np.median(depth[finite])),
                    "max": float(depth[finite].max()),
                    "note": "Depth estimator output is relative/monocular; use ordering and local comparisons, not absolute meters.",
                }
            for hand in phase2.get("hands", []):
                if "wrist_pixel" in hand:
                    hands.append(
                        {
                            "handedness": hand.get("handedness", "unknown"),
                            "wrist_px": hand["wrist_pixel"],
                            "wrist_norm": [
                                round(hand["wrist_pixel"][0] / w, 3),
                                round(hand["wrist_pixel"][1] / h, 3),
                            ],
                            "depth": hand.get("wrist_depth_m"),
                        }
                    )

    def normalize_entity(entity: dict | None) -> dict | None:
        if not entity:
            return None
        cx, cy = entity["centroid_px"]
        x, y, bw, bh = entity["bbox_px"]
        entity["centroid_norm"] = [round(cx / w, 3), round(cy / h, 3)]
        entity["bbox_norm"] = [round(x / w, 3), round(y / h, 3), round(bw / w, 3), round(bh / h, 3)]
        entity["area_frac"] = round(entity["area_px"] / (w * h), 4)
        return entity

    scene = {
        "image": {"width": w, "height": h},
        "detected_entities": {
            "orange_carton_candidate": normalize_entity(orange_obj),
            "dark_mug_or_object_candidate": normalize_entity(dark_obj),
            "skin_hand_region_candidate": normalize_entity(skin_obj),
            "hands_from_phase2": hands,
        },
        "source_depth_summary": depth_summary,
        "layout_hypothesis": {
            "camera": "third-person exocentric, looking at a seated person across a tabletop",
            "target_view": "first-person egocentric view from the seated person's eyes/head camera, looking down at hands and table",
            "preserve": [
                "orange carton near center/front of the interaction",
                "gray/dark mug to the viewer's right of the carton",
                "white tabletop, patterned blue cloth, red mat",
                "both hands visible near the carton",
                "background room objects only if visible from the new viewpoint",
            ],
        },
    }
    return scene


def build_generic_prompt() -> str:
    return (
        "Convert the attached third-person image into a realistic first-person egocentric POV image "
        "from the seated person's eyes. The person is looking down at the table and their hands. "
        "Preserve the same room, table, objects, lighting, and hand-object interaction. Make it look "
        "like a natural wide-angle head-mounted camera photograph, not a drawing."
    )


def build_geometry_prompt(scene: dict) -> str:
    entities = scene["detected_entities"]
    lines = [
        "Convert the attached third-person exocentric image into a realistic first-person egocentric POV image.",
        "",
        "Camera target:",
        "- The new camera is located at the seated person's eyes/head, looking down toward their hands and the tabletop.",
        "- Use a wide-angle head-mounted camera feel, but keep the image photorealistic.",
        "",
        "Geometry/layout constraints from perception:",
    ]
    for name, entity in entities.items():
        if name == "hands_from_phase2":
            continue
        if entity:
            lines.append(f"- {name}: centroid_norm={entity['centroid_norm']}, bbox_norm={entity['bbox_norm']}, area_frac={entity['area_frac']}")
    hands = entities.get("hands_from_phase2") or []
    if hands:
        for hand in hands:
            lines.append(f"- detected {hand['handedness']} wrist at normalized image coordinate {hand['wrist_norm']}")
    if scene.get("source_depth_summary"):
        d = scene["source_depth_summary"]
        lines.append(
            f"- source monocular depth summary: min={d['min']:.3f}, median={d['median']:.3f}, max={d['max']:.3f}; "
            "use this only as relative geometry, not exact metric scale"
        )
    lines += [
        "",
        "Preservation rules:",
        "- Keep the orange carton as the main manipulated object. Do not change it into a different object.",
        "- Keep the dark/gray mug on the right side of the workspace relative to the carton.",
        "- Keep the white table, blue patterned cloth, red mat, and visible small tabletop fixtures.",
        "- Put the person's hands/forearms in the foreground like a true egocentric view.",
        "- Do not invent extra hands, extra cartons, or remove the manipulated carton.",
        "- It is okay to hallucinate unseen regions, but preserve object identities and left/right layout.",
        "",
        "Output one natural, high-resolution, photorealistic ego-view image.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates").mkdir(exist_ok=True)

    rgb = load_rgb(args.image)
    scene = detect_scene(rgb, args.phase2)
    (out / "scene_graph.json").write_text(json.dumps(scene, indent=2))

    generic = build_generic_prompt()
    geometry = build_geometry_prompt(scene)
    (out / "generic_prompt.txt").write_text(generic + "\n")
    (out / "geometry_prompt.txt").write_text(geometry + "\n")

    manifest = {
        "source_image": str(Path(args.image).resolve()),
        "phase2_json": str(Path(args.phase2).resolve()) if args.phase2 else None,
        "prompts": {
            "generic": str((out / "generic_prompt.txt").resolve()),
            "geometry": str((out / "geometry_prompt.txt").resolve()),
        },
        "candidate_slots": [
            "openai_generic",
            "openai_geometry",
            "gemini_generic",
            "gemini_geometry",
            "chatgpt_manual",
            "egoworld",
            "phase3_sparse",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote EgoJudge experiment to {out}")
    print(f"Generic prompt:  {out / 'generic_prompt.txt'}")
    print(f"Geometry prompt: {out / 'geometry_prompt.txt'}")


if __name__ == "__main__":
    main()
