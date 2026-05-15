#!/usr/bin/env python3
"""End-to-end Ego-Exo4D pipeline runner.

Three subcommands, designed to be run in order on the user's machine:

  plan        --frames N
              Pick N evenly-spaced cam01 frames, build a plan of
              (baseline_prompt, wearer_prompt) jobs.

  generate    --plan <path>
              Run OpenAI image-edit for every job. Writes baseline + wearer
              candidates side by side. Skips already-generated outputs.

  eval        --plan <path>
              Compute paired metrics against the matching Aria ego frame
              (the first dataset where we have real paired ego GT).
              Also build a 4-up comparison video:
                exo cam01 | baseline gen | wearer-cond gen | real Aria ego

Cost at N=15: ~30 image calls @ ~$0.04 each = ~$1.20.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import statistics as st
import subprocess
from pathlib import Path
from typing import List

import cv2
import numpy as np

ROOT = Path("experiments/egoexo4d_sfu_cooking025_7")


def cmd_plan(n: int, out_path: Path) -> None:
    cam01_frames = sorted((ROOT / "cam01").glob("frame_*.jpg"))
    aria_frames = sorted((ROOT / "aria01_214-1").glob("frame_*.jpg"))
    if not cam01_frames or not aria_frames:
        raise SystemExit("Frames not extracted yet.")
    total = min(len(cam01_frames), len(aria_frames))
    indices = [int(round(i * (total - 1) / max(n - 1, 1))) for i in range(n)]
    # Deduplicate and keep within range
    seen = []
    for i in indices:
        if i not in seen:
            seen.append(i)
    base_prompt = ROOT / "prompts" / "base_baseline.txt"
    wp_dir = ROOT / "wearer_conditioned_prompts_cam01"
    out_dir_base = ROOT / "generated_baseline_cam01"
    out_dir_wear = ROOT / "generated_wearer_cam01"
    out_dir_base.mkdir(parents=True, exist_ok=True)
    out_dir_wear.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i in seen:
        cam01 = cam01_frames[i]
        aria = aria_frames[i]
        wp_file = wp_dir / f"prompt_{cam01.stem}.txt"
        if not wp_file.exists():
            print(f"  [skip] no wearer prompt for {cam01.stem}")
            continue
        jobs.append(
            {
                "idx": i,
                "exo": str(cam01),
                "aria_gt": str(aria),
                "baseline_prompt": str(base_prompt),
                "wearer_prompt": str(wp_file),
                "out_baseline": str(out_dir_base / f"{cam01.stem}.png"),
                "out_wearer": str(out_dir_wear / f"{cam01.stem}.png"),
            }
        )
    plan = {"n_jobs": len(jobs), "jobs": jobs}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2))
    print(f"wrote {out_path} with {len(jobs)} jobs ({2*len(jobs)} API calls total)")


def _edit_call(client, model, image_path: str, prompt: str, size="1024x1024"):
    """OpenAI image-edit call with input_fidelity fallback."""
    kwargs = dict(model=model, prompt=prompt, size=size, quality="medium")
    with open(image_path, "rb") as f:
        try:
            return client.images.edit(image=f, input_fidelity="high", **kwargs)
        except TypeError:
            pass
    # second attempt without input_fidelity (older SDKs)
    with open(image_path, "rb") as f:
        return client.images.edit(image=f, **kwargs)


def cmd_generate(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set.")
    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("EGOJUDGE_IMAGE_MODEL", "gpt-image-1")
    for job in plan["jobs"]:
        for arm in ("baseline", "wearer"):
            out = Path(job[f"out_{arm}"])
            if out.exists():
                print(f"skip {out.name}")
                continue
            prompt_text = Path(job[f"{arm}_prompt"]).read_text()
            try:
                rsp = _edit_call(client, model, job["exo"], prompt_text)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(base64.b64decode(rsp.data[0].b64_json))
                print(f"  [{arm}] wrote {out.name}")
            except Exception as e:
                print(f"  [{arm}] FAILED {out.name}: {e}")


# ---------------- evaluation ----------------

def _load_rgb(p: str | Path, size=(640, 480)) -> np.ndarray:
    im = cv2.imread(str(p))
    im = cv2.resize(im, size)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _ssim(a, b):
    from scipy.ndimage import uniform_filter

    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    win = 11
    ux, uy = uniform_filter(x, win), uniform_filter(y, win)
    vx = uniform_filter(x * x, win) - ux * ux
    vy = uniform_filter(y * y, win) - uy * uy
    vxy = uniform_filter(x * y, win) - ux * uy
    return float(
        np.mean(((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2) + 1e-12))
    )


def _histw(a, b):
    from scipy.stats import wasserstein_distance

    bins = np.arange(256)
    d = []
    for ch in range(3):
        ha = cv2.calcHist([a], [ch], None, [256], [0, 256]).ravel()
        hb = cv2.calcHist([b], [ch], None, [256], [0, 256]).ravel()
        ha /= max(ha.sum(), 1); hb /= max(hb.sum(), 1)
        d.append(wasserstein_distance(bins, bins, ha, hb))
    return float(np.mean(d))


def _ef1(a, b):
    ea = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 80, 160) > 0
    eb = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), 80, 160) > 0
    inter = float(np.logical_and(ea, eb).sum())
    p = inter / max(float(eb.sum()), 1)
    r = inter / max(float(ea.sum()), 1)
    return float(2 * p * r / max(p + r, 1e-8))


def cmd_eval(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    rows = []
    for job in plan["jobs"]:
        if not Path(job["out_baseline"]).exists() or not Path(job["out_wearer"]).exists():
            continue
        exo = _load_rgb(job["exo"])
        aria = _load_rgb(job["aria_gt"])
        b = _load_rgb(job["out_baseline"])
        w = _load_rgb(job["out_wearer"])
        rows.append(
            {
                "idx": job["idx"],
                "baseline_vs_aria_ssim": _ssim(aria, b),
                "wearer_vs_aria_ssim": _ssim(aria, w),
                "baseline_vs_aria_histw": _histw(aria, b),
                "wearer_vs_aria_histw": _histw(aria, w),
                "baseline_vs_aria_ef1": _ef1(aria, b),
                "wearer_vs_aria_ef1": _ef1(aria, w),
                "baseline_vs_exo_ssim": _ssim(exo, b),
                "wearer_vs_exo_ssim": _ssim(exo, w),
            }
        )
    out = ROOT / "eval_cam01"
    out.mkdir(parents=True, exist_ok=True)
    (out / "paired_metrics.json").write_text(json.dumps(rows, indent=2))

    def summ(rows, key):
        vals = [r[key] for r in rows]
        return f"{st.mean(vals):.3f} +/- {st.stdev(vals):.3f}" if len(vals) > 1 else "n<2"

    summary = {
        "n_paired": len(rows),
        "baseline_vs_aria_ssim": summ(rows, "baseline_vs_aria_ssim"),
        "wearer_vs_aria_ssim": summ(rows, "wearer_vs_aria_ssim"),
        "baseline_vs_aria_histw": summ(rows, "baseline_vs_aria_histw"),
        "wearer_vs_aria_histw": summ(rows, "wearer_vs_aria_histw"),
        "baseline_vs_aria_ef1": summ(rows, "baseline_vs_aria_ef1"),
        "wearer_vs_aria_ef1": summ(rows, "wearer_vs_aria_ef1"),
        "baseline_vs_exo_ssim": summ(rows, "baseline_vs_exo_ssim"),
        "wearer_vs_exo_ssim": summ(rows, "wearer_vs_exo_ssim"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Build 4-up comparison video at 2 fps over the rows we have.
    TARGET = (480, 360)
    hdr = 30
    frames_video = out / "comparison_frames"
    frames_video.mkdir(parents=True, exist_ok=True)
    for r in rows:
        job = next(j for j in plan["jobs"] if j["idx"] == r["idx"])
        exo = cv2.resize(cv2.imread(job["exo"]), TARGET)
        b = cv2.resize(cv2.imread(job["out_baseline"]), TARGET)
        w = cv2.resize(cv2.imread(job["out_wearer"]), TARGET)
        aria = cv2.resize(cv2.imread(job["aria_gt"]), TARGET)
        gap_v = np.full((TARGET[1], 6, 3), 240, np.uint8)
        gap_h = np.full((hdr, 4 * TARGET[0] + 3 * 6, 3), 245, np.uint8)
        for i, label in enumerate(["cam01 (exo)", "baseline ego", "wearer-cond ego", "Aria ego GT"]):
            cv2.putText(gap_h, label, (i * (TARGET[0] + 6) + 8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
        strip = np.concatenate([exo, gap_v, b, gap_v, w, gap_v, aria], axis=1)
        frame = np.concatenate([gap_h, strip], axis=0)
        cv2.imwrite(str(frames_video / f"frame_{r['idx']:03d}.jpg"), frame)
    # ffmpeg into mp4
    video_out = ROOT / "clips" / "comparison_4up.mp4"
    video_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", "2",
        "-pattern_type", "glob",
        "-i", str(frames_video / "frame_*.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", "30",
        str(video_out),
    ]
    subprocess.run(cmd, check=False)
    print(f"wrote {video_out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("plan")
    p1.add_argument("--n", type=int, default=15)
    p1.add_argument("--out", default=str(ROOT / "pipeline_plan.json"))
    p2 = sub.add_parser("generate")
    p2.add_argument("--plan", default=str(ROOT / "pipeline_plan.json"))
    p3 = sub.add_parser("eval")
    p3.add_argument("--plan", default=str(ROOT / "pipeline_plan.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "plan":
        cmd_plan(args.n, Path(args.out))
    elif args.cmd == "generate":
        cmd_generate(Path(args.plan))
    elif args.cmd == "eval":
        cmd_eval(Path(args.plan))


if __name__ == "__main__":
    main()
