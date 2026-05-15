#!/usr/bin/env python3
"""Run the real GroundingDINO + SAM2 mask-conditioned exo-to-ego path.

This wrapper keeps a clean experiment folder:

  source image
    -> GroundingDINO boxes + SAM2 masks
    -> mask-composed ego condition
    -> optional OpenAI image edit generation
    -> SAM2 masks on generated candidates
    -> mask-level constraint score
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Source exocentric frame.")
    p.add_argument("--out", required=True, help="Clean output directory.")
    p.add_argument("--device", default="mps")
    p.add_argument("--box-threshold", type=float, default=0.12)
    p.add_argument("--generate", action="store_true", help="Call the OpenAI image edit API for the mask condition.")
    p.add_argument("--image-model", default="gpt-image-2")
    p.add_argument("--size", default="1024x1024")
    p.add_argument("--quality", default="medium")
    p.add_argument("--candidate", action="append", default=[], help="Existing generated ego image to segment and score.")
    p.add_argument("--candidate-name", action="append", default=[], help="Name for the corresponding --candidate.")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    source_segments = out / "source_segments"
    condition = out / "mask_condition"
    candidates_root = out / "candidate_segments"
    score_dir = out / "score"

    run([
        sys.executable,
        "scripts/grounded_sam2_segment.py",
        "--image",
        args.image,
        "--out",
        str(source_segments),
        "--device",
        args.device,
        "--box-threshold",
        str(args.box_threshold),
    ])
    run([
        sys.executable,
        "scripts/mask_condition_render.py",
        "--image",
        args.image,
        "--segments",
        str(source_segments / "segments.json"),
        "--out",
        str(condition),
    ])

    candidate_paths = [Path(p) for p in args.candidate]
    candidate_names = list(args.candidate_name)
    if candidate_names and len(candidate_names) != len(candidate_paths):
        raise SystemExit("--candidate-name must be provided once per --candidate")
    if not candidate_names:
        candidate_names = [p.stem for p in candidate_paths]

    if args.generate:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is missing; rerun without --generate or add it to the environment/.env")
        generated = out / "generated" / "mask_condition_gpt_image_2.png"
        run([
            sys.executable,
            "scripts/egojudge_openai.py",
            "--image",
            str(condition / "mask_ego_condition.png"),
            "--prompt",
            str(condition / "mask_condition_prompt.txt"),
            "--out",
            str(generated),
            "--model",
            args.image_model,
            "--size",
            args.size,
            "--quality",
            args.quality,
            "--input-fidelity",
            "high",
        ])
        candidate_paths.append(generated)
        candidate_names.append("mask_condition")

    segment_paths: list[Path] = []
    for name, image in zip(candidate_names, candidate_paths):
        seg_dir = candidates_root / name
        run([
            sys.executable,
            "scripts/grounded_sam2_segment.py",
            "--image",
            str(image),
            "--out",
            str(seg_dir),
            "--device",
            args.device,
            "--box-threshold",
            str(args.box_threshold),
        ])
        segment_paths.append(seg_dir / "segments.json")

    if segment_paths:
        run([
            sys.executable,
            "scripts/mask_constraint_score.py",
            "--source-segments",
            str(source_segments / "segments.json"),
            "--candidate-segments",
            *[str(p) for p in segment_paths],
            "--names",
            *candidate_names,
            "--out",
            str(score_dir),
        ])

    print(f"Wrote clean SAM2 mask-conditioned run to {out}")


if __name__ == "__main__":
    main()
