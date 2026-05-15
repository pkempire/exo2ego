#!/usr/bin/env python3
"""Best-of-K candidate generation with diverse prompt mutations.

For a given exocentric input we generate K candidates per frame using a small
registry of prompt-mutation axes:

  - seed:           same prompt, K different seeds
  - camera:         vary the implied camera placement
  - style_anchor:   vary the implied camera *brand* (carries geometric prior)
  - verbosity:      short prompt vs full scene graph
  - hybrid:         a few mutations combined

EgoJudge then picks the highest-scoring candidate from the K. We log the
per-candidate score and the within-set spread so we can report:

  single_shot_score   = score of the first candidate
  best_of_K_score     = max over K
  best_of_K_gap       = best - single_shot

Like the prompt-following diagnostic this script is offline-safe: without an
OPENAI key it only writes the plan and the prompts.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Iterable


CAMERA_VARIANTS = [
    "a head-mounted GoPro at chest level looking down",
    "a head-mounted Aria glasses camera looking forward and slightly down",
    "an iPhone clipped to a baseball cap brim, pointed at the table",
    "a phone mounted at chest level, table-edge POV",
    "a body-cam at sternum height, gentle downward tilt",
]

STYLE_ANCHORS = [
    "GoPro Hero footage, wide field of view",
    "Aria glasses POV, slight fish-eye distortion",
    "iPhone 15 selfie cam, natural color",
    "DSLR with 35 mm lens",
    "ZV-E10 mirrorless, soft daylight",
]

VERBOSITY_VARIANTS = [
    "Convert this third-person frame into a first-person tabletop POV showing the same hands and objects.",
    "{base_scene_graph}",
]


def build_prompts(
    scene_graph_path: str | None,
    base_prompt: str,
    k: int,
) -> list[dict]:
    sg = ""
    if scene_graph_path and Path(scene_graph_path).exists():
        sg = Path(scene_graph_path).read_text()
    prompts = []
    for i in range(k):
        cam = CAMERA_VARIANTS[i % len(CAMERA_VARIANTS)]
        style = STYLE_ANCHORS[i % len(STYLE_ANCHORS)]
        text = (
            f"{base_prompt}\n\n"
            f"Target camera: {cam}.\n"
            f"Visual style: {style}.\n"
            "Preserve hand count, manipulated objects, contact, and overall "
            "scene layout. No extra hands, no extra people, no debug panels."
        )
        if sg:
            text = sg + "\n\n" + text
        prompts.append(
            {
                "k": i,
                "camera": cam,
                "style": style,
                "prompt": text,
            }
        )
    return prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("--exo", required=True, help="Exo image path.")
    p1.add_argument("--scene-graph", default=None)
    p1.add_argument("--base-prompt", required=True, help="Base prompt file.")
    p1.add_argument("--k", type=int, default=5)
    p1.add_argument("--out", required=True)

    p2 = sub.add_parser("generate")
    p2.add_argument("--plan", required=True)

    return p.parse_args()


def prepare(args) -> None:
    base = Path(args.base_prompt).read_text()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts(args.scene_graph, base, args.k)
    plan = {
        "exo": args.exo,
        "scene_graph": args.scene_graph,
        "k": args.k,
        "candidates": [
            {**p, "out": str(out / f"cand_k{p['k']}.png")} for p in prompts
        ],
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    for p in plan["candidates"]:
        Path(out / f"prompt_k{p['k']}.txt").write_text(p["prompt"])
    print(f"wrote plan with {args.k} candidates to {out / 'plan.json'}")


def run_generate(args) -> None:
    plan = json.loads(Path(args.plan).read_text())
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; not generating.")
        return
    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("EGOJUDGE_IMAGE_MODEL", "gpt-image-1")

    def _edit(image_file, prompt):
        kwargs = dict(
            model=model,
            image=image_file,
            prompt=prompt,
            size="1536x1024",
            quality="medium",
        )
        try:
            return client.images.edit(**kwargs, input_fidelity="high")
        except TypeError:
            return client.images.edit(**kwargs)

    for c in plan["candidates"]:
        out = Path(c["out"])
        if out.exists():
            print(f"skip {out.name}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(plan["exo"], "rb") as f:
            rsp = _edit(f, c["prompt"])
        out.write_bytes(base64.b64decode(rsp.data[0].b64_json))
        print(f"wrote {out}")


def main() -> None:
    args = parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "generate":
        run_generate(args)


if __name__ == "__main__":
    main()
