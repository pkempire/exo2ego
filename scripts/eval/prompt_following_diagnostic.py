#!/usr/bin/env python3
"""Prompt-Following Diagnostic.

This script answers the question: when we put a normalized coordinate into the
prompt, does the image generator actually place the object there?

We test this by asking the model to place a single red mug at a grid of
positions. We then detect the mug in each output by HSV thresholding and
compare the detected centroid to the requested coordinate.

We also run a second pass with qualitative wording ("upper-right",
"center-left", ...) instead of numbers, so we can compare numeric vs
qualitative grounding side by side.

The script is OFFLINE-AWARE: if no API key is configured, it only writes the
prompt set and the analysis stub, so you can plan the run on a machine without
the API.

Usage:
    # 1. Build prompt set + base scene + detection script (no API calls)
    python scripts/prompt_following_diagnostic.py prepare \\
        --out experiments/prompt_following

    # 2. Run generation (needs OPENAI_API_KEY)
    python scripts/prompt_following_diagnostic.py generate \\
        --plan experiments/prompt_following/plan.json

    # 3. Score the outputs against expected positions
    python scripts/prompt_following_diagnostic.py score \\
        --plan experiments/prompt_following/plan.json
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


GRID = [0.15, 0.35, 0.50, 0.65, 0.85]
QUAL_LABEL_GRID = {
    0.15: "left edge",
    0.35: "left of center",
    0.50: "center",
    0.65: "right of center",
    0.85: "right edge",
}
QUAL_VERT_GRID = {
    0.15: "near the top",
    0.35: "upper half",
    0.50: "middle",
    0.65: "lower half",
    0.85: "near the bottom",
}


def make_blank_scene(out_path: Path, size: tuple[int, int] = (1024, 768)) -> None:
    """Render a clean tabletop background to act as the base scene."""
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    # plaster wall in upper half
    img[: int(h * 0.4)] = (235, 230, 225)
    # wooden tabletop in lower half (just a warm gradient)
    for y in range(int(h * 0.4), h):
        t = (y - h * 0.4) / (h - h * 0.4)
        img[y] = (
            int(135 + 30 * (1 - t)),
            int(100 + 20 * (1 - t)),
            int(70 + 10 * (1 - t)),
        )[::-1]
    # subtle horizon line
    cv2.line(img, (0, int(h * 0.4)), (w, int(h * 0.4)), (200, 195, 190), 2)
    cv2.imwrite(str(out_path), img)


def make_prompt_numeric(x: float, y: float) -> str:
    return (
        "Photorealistic first-person view of an empty tabletop. "
        "Place a single solid red ceramic mug on the table. "
        f"The mug's center must appear at normalized image coordinates "
        f"(x={x:.2f}, y={y:.2f}), where (0, 0) is the top-left corner "
        "and (1, 1) is the bottom-right corner of the output image. "
        "Do not add any other objects. Soft, even daylight. No text or labels."
    )


def make_prompt_qualitative(x: float, y: float) -> str:
    horiz = QUAL_LABEL_GRID[x]
    vert = QUAL_VERT_GRID[y]
    return (
        "Photorealistic first-person view of an empty tabletop. "
        f"Place a single solid red ceramic mug on the table, {horiz} of the "
        f"frame and {vert} of the frame. "
        "Do not add any other objects. Soft, even daylight. No text or labels."
    )


def build_plan(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "base_scene.png"
    make_blank_scene(base)
    cells = []
    for y in GRID:
        for x in GRID:
            cells.append(
                {
                    "x": x,
                    "y": y,
                    "numeric_prompt": make_prompt_numeric(x, y),
                    "qualitative_prompt": make_prompt_qualitative(x, y),
                    "numeric_out": str(
                        out_dir / "numeric" / f"mug_{x:.2f}_{y:.2f}.png"
                    ),
                    "qualitative_out": str(
                        out_dir / "qualitative" / f"mug_{x:.2f}_{y:.2f}.png"
                    ),
                }
            )
    (out_dir / "numeric").mkdir(exist_ok=True)
    (out_dir / "qualitative").mkdir(exist_ok=True)
    plan = {"base_scene": str(base), "cells": cells, "grid": GRID}
    plan_path = out_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    return plan_path


def detect_red_mug(image_path: str | Path) -> tuple[float, float] | None:
    """Return normalized (x, y) of the red mug, or None if not found."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # red is wrapping the hue axis, so use two ranges
    m1 = cv2.inRange(hsv, np.array([0, 110, 70]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 110, 70]), np.array([180, 255, 255]))
    m = cv2.bitwise_or(m1, m2)
    # find largest red blob
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 50:
        return None
    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"] / w
    cy = M["m01"] / M["m00"] / h
    return float(cx), float(cy)


def run_generation(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    cells = plan["cells"]
    base = plan["base_scene"]
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY not set. Skipping generation; use this script's "
            "output directory once you run it on your machine."
        )
        return
    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("EGOJUDGE_IMAGE_MODEL", "gpt-image-1")

    def _edit(image_file, prompt):
        """Call client.images.edit with graceful fallback for older SDKs."""
        kwargs = dict(
            model=model,
            image=image_file,
            prompt=prompt,
            size="1024x1024",
            quality="medium",
        )
        try:
            return client.images.edit(**kwargs, input_fidelity="high")
        except TypeError:
            return client.images.edit(**kwargs)

    for cell in cells:
        for kind in ("numeric", "qualitative"):
            out = Path(cell[f"{kind}_out"])
            if out.exists():
                print(f"skip {out.name}")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(base, "rb") as f:
                rsp = _edit(f, cell[f"{kind}_prompt"])
            data = base64.b64decode(rsp.data[0].b64_json)
            out.write_bytes(data)
            print(f"wrote {out}")


def score(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    rows = []
    for cell in plan["cells"]:
        for kind in ("numeric", "qualitative"):
            out = Path(cell[f"{kind}_out"])
            det = detect_red_mug(out) if out.exists() else None
            row = {
                "kind": kind,
                "x_req": cell["x"],
                "y_req": cell["y"],
                "x_det": det[0] if det else None,
                "y_det": det[1] if det else None,
                "found": det is not None,
            }
            if det:
                dx = det[0] - cell["x"]
                dy = det[1] - cell["y"]
                row["l2_error"] = float(math.sqrt(dx * dx + dy * dy))
            rows.append(row)
    out_dir = plan_path.parent
    (out_dir / "scores.json").write_text(json.dumps({"rows": rows}, indent=2))
    # summary
    summary = {}
    for kind in ("numeric", "qualitative"):
        sub = [r for r in rows if r["kind"] == kind and r["found"]]
        if not sub:
            summary[kind] = {"n": 0}
            continue
        x_req = np.array([r["x_req"] for r in sub])
        y_req = np.array([r["y_req"] for r in sub])
        x_det = np.array([r["x_det"] for r in sub])
        y_det = np.array([r["y_det"] for r in sub])
        l2 = np.array([r["l2_error"] for r in sub])
        rx = float(np.corrcoef(x_req, x_det)[0, 1]) if x_req.std() > 0 else 0
        ry = float(np.corrcoef(y_req, y_det)[0, 1]) if y_req.std() > 0 else 0
        summary[kind] = {
            "n": len(sub),
            "mean_l2": float(l2.mean()),
            "median_l2": float(np.median(l2)),
            "pearson_x": rx,
            "pearson_y": ry,
        }
    summary["found_rate_numeric"] = (
        sum(1 for r in rows if r["kind"] == "numeric" and r["found"]) / 25
    )
    summary["found_rate_qualitative"] = (
        sum(1 for r in rows if r["kind"] == "qualitative" and r["found"]) / 25
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("generate")
    p2.add_argument("--plan", required=True)
    p3 = sub.add_parser("score")
    p3.add_argument("--plan", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "prepare":
        plan = build_plan(Path(args.out))
        print(f"wrote plan: {plan}")
        print(f"   25 cells x 2 conditions = 50 generations")
    elif args.cmd == "generate":
        run_generation(Path(args.plan))
    elif args.cmd == "score":
        score(Path(args.plan))


if __name__ == "__main__":
    main()
