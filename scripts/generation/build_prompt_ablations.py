#!/usr/bin/env python3
"""Create controlled prompt ablations for exo-to-ego generation.

The goal is to change one factor at a time so we can benchmark whether camera
hints, scene graph details, and failure-risk negatives actually help.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-graph", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame-id", default="unknown")
    parser.add_argument("--style", default="low-light phone/chest-mounted POV")
    return parser.parse_args()


def load_scene(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def camera_lines(scene: dict) -> list[str]:
    camera = scene.get("camera") or {}
    lines = [
        "Transform the attached exocentric manipulation image into the matching first-person egocentric view.",
        "The target should look like a real head/chest/phone-mounted POV camera looking down at the same moment.",
    ]
    if camera.get("body_side_in_target"):
        lines.append(f"Put the body-side/near workspace toward the {camera['body_side_in_target']} of the output image.")
    if camera.get("far_side_in_target"):
        lines.append(f"Put the far side of the workspace toward the {camera['far_side_in_target']} of the output image.")
    if camera.get("likely_target_ego_view"):
        lines.append(f"Target view: {camera['likely_target_ego_view']}")
    return lines


def compact_scene(scene: dict, include_camera: bool = True, include_failures: bool = True) -> dict:
    if not scene:
        return {}
    keep = {
        "image_summary": scene.get("image_summary"),
        "workspace": scene.get("workspace"),
        "hands": scene.get("hands", []),
        "objects": scene.get("objects", []),
        "spatial_constraints": scene.get("spatial_constraints", []),
    }
    if include_camera:
        keep["camera"] = scene.get("camera", {})
    if include_failures:
        keep["failure_risks"] = scene.get("failure_risks", [])
    return keep


def generic_prompt(style: str) -> str:
    return "\n".join(
        [
            "Transform the attached third-person manipulation image into a realistic first-person egocentric camera view.",
            f"Make it look like a real {style} frame, not a studio product photo.",
            "Preserve the same task, hands, objects, lighting, and moment as much as possible.",
            "Do not add faces or identifying features.",
            "Output one natural photorealistic first-person image.",
        ]
    )


def safe_prompt(style: str) -> str:
    return "\n".join(
        [
            "Transform the attached overhead/tabletop image into a matching first-person camera view of the same scene.",
            "",
            "Camera/view:",
            "- The output should look like a real first-person action-camera or phone POV frame looking downward at the workspace.",
            "- The near/body-side edge of the workspace should be near the bottom of the image.",
            "- Hands and forearms should enter from the lower or side image edges.",
            f"- Preserve the source lighting and camera quality: {style}.",
            "",
            "Preservation constraints:",
            "- Preserve all visible manipulation-relevant objects.",
            "- Preserve the same moment and object layout as much as possible.",
            "- Do not create a studio scene, fantasy render, clean product photo, or marketing image.",
            "- Do not add faces, identifying features, extra hands, or unrelated objects.",
            "",
            "Output one natural photorealistic first-person image.",
        ]
    )


def scene_prompt(scene: dict, style: str, include_camera: bool = True, include_failures: bool = True, include_scene: bool = True) -> str:
    lines = []
    if include_camera:
        lines.extend(camera_lines(scene))
    else:
        lines.extend(
            [
                "Transform the attached exocentric manipulation image into a realistic first-person egocentric view.",
                "Use a plausible embodied camera angle while preserving the same task moment.",
            ]
        )
    lines.extend(
        [
            "",
            f"Camera style: {style}. Preserve source lighting and image quality.",
            "Do not beautify the scene or change it into a studio/product render.",
        ]
    )
    if include_scene and scene:
        lines.extend(
            [
                "",
                "Use this extracted scene graph as hard conditioning:",
                json.dumps(compact_scene(scene, include_camera=include_camera, include_failures=include_failures), indent=2),
            ]
        )
    lines.extend(
        [
            "",
            "Generation requirements:",
            "- Preserve visible hands/forearms and manipulation-relevant objects.",
            "- Preserve left/right and near/far spatial relationships.",
            "- Preserve hand-object contact state and action phase.",
            "- If unseen regions are required, hallucinate plausibly without moving the main objects.",
            "- Output one natural photorealistic first-person image.",
        ]
    )
    if not include_failures:
        return "\n".join(lines)
    failures = scene.get("failure_risks", []) if scene else []
    if failures:
        lines.extend(["", "Avoid these likely failures:"] + [f"- {risk}" for risk in failures])
    else:
        lines.append("- Do not add extra hands, unrelated objects, or faces.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    scene = load_scene(args.scene_graph)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompts = {
        "generic": generic_prompt(args.style),
        "safe": safe_prompt(args.style),
        "scene_full": scene_prompt(scene, args.style, include_camera=True, include_failures=True, include_scene=True),
        "scene_no_camera": scene_prompt(scene, args.style, include_camera=False, include_failures=True, include_scene=True),
        "scene_no_failure_risks": scene_prompt(scene, args.style, include_camera=True, include_failures=False, include_scene=True),
        "camera_only": scene_prompt(scene, args.style, include_camera=True, include_failures=True, include_scene=False),
    }
    manifest = {"frame_id": args.frame_id, "scene_graph": str(Path(args.scene_graph).resolve()) if args.scene_graph else None, "prompts": {}}
    for name, text in prompts.items():
        path = out / f"{name}.txt"
        path.write_text(text.strip() + "\n")
        manifest["prompts"][name] = str(path.resolve())
    (out / "prompt_ablation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(prompts)} prompt ablations to {out}")
    print(f"Wrote {out / 'prompt_ablation_manifest.json'}")


if __name__ == "__main__":
    main()
