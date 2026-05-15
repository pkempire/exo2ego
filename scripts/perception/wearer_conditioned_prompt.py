#!/usr/bin/env python3
"""Build a wearer-conditioned prompt from a wearer_pose.json record.

Strategy: where the head/eye and torso are detected in the exo image, we
emit a structured "where the ego camera is and which way it looks" block,
plus a qualitative wording block (which we believe T2I models follow more
reliably than numeric coordinates --- see paper section 8).

Usage:
    python scripts/wearer_conditioned_prompt.py \\
        --wearer experiments/.../wearer_pose/wearer_pose.json \\
        --base-prompt experiments/.../prompts/objects_camera_sanitized.txt \\
        --out experiments/.../wearer_conditioned_prompts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def qual_x(x: float) -> str:
    if x < 0.25:
        return "the left side"
    if x < 0.45:
        return "the left-of-center"
    if x < 0.55:
        return "the center"
    if x < 0.75:
        return "the right-of-center"
    return "the right side"


def qual_y(y: float) -> str:
    if y < 0.25:
        return "near the top"
    if y < 0.45:
        return "the upper half"
    if y < 0.55:
        return "vertical middle"
    if y < 0.75:
        return "the lower half"
    return "near the bottom"


def head_orientation_phrase(orientation: str) -> str:
    return {
        "frontal": "facing the exo camera",
        "left_profile": "in left profile (looking to viewer's left)",
        "right_profile": "in right profile (looking to viewer's right)",
        "back": "back of head toward the exo camera",
        "unknown": "of unknown orientation",
    }.get(orientation or "unknown", "of unknown orientation")


def build_block(row: dict) -> str:
    if not row.get("wearer_visible"):
        return ""
    lines = ["", "Wearer / ego-camera anchor (extracted from the exo frame):"]
    ec = row.get("implied_eye_norm")
    if ec:
        lines.append(
            f"- The wearer's head/eye region is at {qual_x(ec[0])} of the frame, "
            f"{qual_y(ec[1])} (approx normalized {ec[0]:.2f}, {ec[1]:.2f})."
        )
    pose = row.get("pose")
    if pose:
        tc = pose.get("torso_center_norm")
        if tc:
            lines.append(
                f"- The wearer's torso is at {qual_x(tc[0])} of the frame, "
                f"around the {qual_y(tc[1])}."
            )
    if row.get("head_orientation"):
        lines.append(f"- The head is {head_orientation_phrase(row['head_orientation'])}.")
    if row.get("body_side"):
        lines.append(f"- Body relative position in the frame: {row['body_side']}.")
    lines.append(
        "- Therefore the egocentric viewpoint should be positioned at the "
        "wearer's head, looking roughly in the direction the head is facing, "
        "and the wearer's own arms should enter the ego frame from below."
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wearer", required=True)
    p.add_argument("--base-prompt", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rec = json.loads(Path(args.wearer).read_text())
    base = Path(args.base_prompt).read_text()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rec.get("rows", []):
        if not row.get("wearer_visible"):
            continue
        frame_path = row["frame"]
        stem = Path(frame_path).stem
        block = build_block(row)
        text = base.rstrip() + "\n" + block + "\n"
        prompt_file = out / f"prompt_{stem}.txt"
        prompt_file.write_text(text)
        manifest.append({
            "frame": frame_path,
            "prompt": str(prompt_file),
            "wearer_visible": True,
        })
    (out / "manifest.json").write_text(json.dumps({"rows": manifest}, indent=2))
    print(f"wrote {len(manifest)} wearer-conditioned prompts to {out}")


if __name__ == "__main__":
    main()
