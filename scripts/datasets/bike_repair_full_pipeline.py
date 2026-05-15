#!/usr/bin/env python3
"""Full pipeline runner for Ego-Exo4D bike-repair take (cmu_bike08_2, cam04).

Three arms to compare on the same N frames against the real Aria ego GT:
    arm 0: baseline prompt-only (text only)
    arm 1: wearer-conditioned prompt (text + Pose-derived wearer anchor)
    arm 2: EgoWorld-style sparse-RGB inpaint (depth backprojection + inpaint)

Subcommands:
    plan      --n N         build the plan (no API)
    generate  --plan PATH   run all 3 arms for every frame  (~3N API calls)
    eval      --plan PATH   compute paired metrics vs Aria GT, build 5-up video
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

import cv2
import numpy as np

ROOT = Path("experiments/egoexo4d_cmu_bike08_2")


def cmd_plan(n: int, out_path: Path) -> None:
    cam_dir = ROOT / "cam04"
    aria_dir = ROOT / "aria01_214-1"
    exo_frames = sorted(cam_dir.glob("frame_*.jpg"))
    aria_frames = sorted(aria_dir.glob("frame_*.jpg"))
    total = min(len(exo_frames), len(aria_frames))
    indices = [int(round(i * (total - 1) / max(n - 1, 1))) for i in range(n)]
    seen = []
    for i in indices:
        if i not in seen: seen.append(i)

    # base prompt
    base_prompt_path = ROOT / "prompts" / "bike_base.txt"
    base_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    base_prompt_path.write_text(
        "Transform this third-person bike-repair-shop frame into a realistic "
        "first-person view from a head-mounted Aria-style camera worn by the "
        "mechanic visible in the frame. The mechanic is leaning over a bike "
        "on a repair stand; the egocentric camera is at their eye level, "
        "looking down at the bike's frame, fork, or wheels where their hands "
        "are working. Preserve the surrounding bike shop (tool wall, hanging "
        "wheels, other bikes), preserve the specific bike geometry, preserve "
        "hand positions and the tool being used. Render with a mild fish-eye "
        "look and the wearer's arms entering the ego frame from the bottom. "
        "No extra hands, no extra people, no new tools.\n"
    )

    # wearer-conditioned prompts (per-frame anchor block from Pose)
    import json as _json
    wp = _json.loads((ROOT / "wearer_pose_cam04" / "wearer_pose.json").read_text())
    by_frame = {Path(r["frame"]).stem: r for r in wp.get("rows", []) if r.get("wearer_visible")}
    wp_dir = ROOT / "wearer_conditioned_prompts_cam04"
    wp_dir.mkdir(parents=True, exist_ok=True)
    base = base_prompt_path.read_text()

    def qual_x(x):
        if x < 0.25: return "the left side"
        if x < 0.45: return "the left-of-center"
        if x < 0.55: return "the center"
        if x < 0.75: return "the right-of-center"
        return "the right side"
    def qual_y(y):
        if y < 0.25: return "near the top"
        if y < 0.45: return "the upper half"
        if y < 0.55: return "vertical middle"
        if y < 0.75: return "the lower half"
        return "near the bottom"

    for stem, row in by_frame.items():
        ec = row.get("implied_eye_norm") or [0.5, 0.5]
        body = row.get("body_side") or "center"
        orient = row.get("head_orientation") or "unknown"
        block = (
            "\n\nWearer / ego-camera anchor (extracted from the exo frame):\n"
            f"- The mechanic's head/eye region is at {qual_x(ec[0])} of the frame, {qual_y(ec[1])}.\n"
            f"- Body relative position in the frame: {body}.\n"
            f"- Head orientation: {orient}.\n"
            "- Therefore the egocentric viewpoint should be positioned at the "
            "mechanic's head, looking forward and slightly down toward the "
            "bike, with hands entering the ego frame from below.\n"
        )
        (wp_dir / f"prompt_{stem}.txt").write_text(base + block)

    jobs = []
    out_b = ROOT / "generated_baseline_cam04"; out_b.mkdir(parents=True, exist_ok=True)
    out_w = ROOT / "generated_wearer_cam04"; out_w.mkdir(parents=True, exist_ok=True)
    out_e = ROOT / "generated_egoworld_cam04"; out_e.mkdir(parents=True, exist_ok=True)
    sparse_dir = ROOT / "egoworld_sparse_cam04"; sparse_dir.mkdir(parents=True, exist_ok=True)
    for i in seen:
        exo = exo_frames[i]
        aria = aria_frames[i]
        wp_file = wp_dir / f"prompt_{exo.stem}.txt"
        if not wp_file.exists():
            continue
        jobs.append({
            "idx": i,
            "exo": str(exo),
            "aria_gt": str(aria),
            "baseline_prompt": str(base_prompt_path),
            "wearer_prompt": str(wp_file),
            "out_baseline": str(out_b / f"{exo.stem}.png"),
            "out_wearer":   str(out_w / f"{exo.stem}.png"),
            "out_egoworld": str(out_e / f"{exo.stem}.png"),
            "sparse_dir":   str(sparse_dir / exo.stem),
        })
    plan = {"n_jobs": len(jobs), "jobs": jobs}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2))
    print(f"wrote {out_path}  with {len(jobs)} jobs ({3*len(jobs)} API calls total)")


def _edit_call(client, model, image_path, prompt, size, mask_path=None):
    kw = dict(model=model, prompt=prompt, size=size, quality="medium")
    with open(image_path, "rb") as f:
        if mask_path:
            with open(mask_path, "rb") as fm:
                try: return client.images.edit(image=f, mask=fm, input_fidelity="high", **kw)
                except TypeError: pass
        else:
            try: return client.images.edit(image=f, input_fidelity="high", **kw)
            except TypeError: pass
    with open(image_path, "rb") as f:
        if mask_path:
            with open(mask_path, "rb") as fm:
                return client.images.edit(image=f, mask=fm, **kw)
        return client.images.edit(image=f, **kw)


def cmd_generate(plan_path: Path, skip_egoworld: bool = False) -> None:
    plan = json.loads(plan_path.read_text())
    try:
        from dotenv import load_dotenv; load_dotenv()
    except Exception: pass
    if not os.environ.get("OPENAI_API_KEY"): raise SystemExit("OPENAI_API_KEY not set.")
    from openai import OpenAI
    client = OpenAI()
    model = os.environ.get("EGOJUDGE_IMAGE_MODEL", "gpt-image-1")

    # Pre-build sparse RGB for every job
    if not skip_egoworld:
        for job in plan["jobs"]:
            sd = Path(job["sparse_dir"])
            if (sd / "hole_mask_rgba.png").exists():
                continue
            print(f"build sparse for idx {job['idx']:03d}")
            subprocess.run([
                "python3", "scripts/egoworld_pathway.py",
                "--exo", job["exo"],
                "--out", str(sd),
                "--skip-inpaint",
            ], check=False)

    for job in plan["jobs"]:
        # Arm 0: baseline
        out = Path(job["out_baseline"])
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            rsp = _edit_call(client, model, job["exo"], Path(job["baseline_prompt"]).read_text(), "1536x1024")
            out.write_bytes(base64.b64decode(rsp.data[0].b64_json))
            print(f"  baseline  idx {job['idx']:03d}")
        # Arm 1: wearer
        out = Path(job["out_wearer"])
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            rsp = _edit_call(client, model, job["exo"], Path(job["wearer_prompt"]).read_text(), "1536x1024")
            out.write_bytes(base64.b64decode(rsp.data[0].b64_json))
            print(f"  wearer    idx {job['idx']:03d}")
        # Arm 2: EgoWorld-style sparse-RGB inpaint
        if not skip_egoworld:
            out = Path(job["out_egoworld"])
            if not out.exists():
                sparse_rgb = Path(job["sparse_dir"]) / "sparse_ego_rgb.png"
                sparse_mask = Path(job["sparse_dir"]) / "hole_mask_rgba.png"
                if not sparse_rgb.exists() or not sparse_mask.exists():
                    print(f"  egoworld  idx {job['idx']:03d}  SKIP (no sparse)"); continue
                out.parent.mkdir(parents=True, exist_ok=True)
                prompt = (
                    "This is a sparse egocentric (head-mounted) view of a bike-repair "
                    "shop, reprojected from a third-person frame. Transparent pixels "
                    "must be filled. Preserve every visible pixel exactly. Fill the "
                    "transparent regions plausibly so the result looks like a real "
                    "Aria glasses ego frame of a mechanic working on a bike: same "
                    "shop, same bike, same tool wall, hands and arms entering the "
                    "ego frame from below if appropriate."
                )
                rsp = _edit_call(client, model, sparse_rgb, prompt, "1024x1024", mask_path=sparse_mask)
                out.write_bytes(base64.b64decode(rsp.data[0].b64_json))
                print(f"  egoworld  idx {job['idx']:03d}")


def _load(p, size=(640, 480)):
    im = cv2.resize(cv2.imread(str(p)), size)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _ssim(a, b):
    from scipy.ndimage import uniform_filter
    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    ux, uy = uniform_filter(x, 11), uniform_filter(y, 11)
    vx = uniform_filter(x*x, 11) - ux*ux
    vy = uniform_filter(y*y, 11) - uy*uy
    vxy = uniform_filter(x*y, 11) - ux*uy
    return float(np.mean(((2*ux*uy + c1)*(2*vxy + c2)) / ((ux*ux + uy*uy + c1)*(vx + vy + c2) + 1e-12)))


def _histw(a, b):
    from scipy.stats import wasserstein_distance
    bins = np.arange(256); ds = []
    for ch in range(3):
        ha = cv2.calcHist([a], [ch], None, [256], [0,256]).ravel()
        hb = cv2.calcHist([b], [ch], None, [256], [0,256]).ravel()
        ha /= max(ha.sum(),1); hb /= max(hb.sum(),1)
        ds.append(wasserstein_distance(bins, bins, ha, hb))
    return float(np.mean(ds))


def cmd_eval(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    rows = []
    for job in plan["jobs"]:
        bp = Path(job["out_baseline"]); wp = Path(job["out_wearer"]); ep = Path(job["out_egoworld"])
        if not bp.exists() or not wp.exists(): continue
        exo = _load(job["exo"]); aria = _load(job["aria_gt"]); b = _load(bp); w = _load(wp)
        row = {
            "idx": job["idx"],
            "baseline_vs_aria_ssim": _ssim(aria, b),
            "wearer_vs_aria_ssim":   _ssim(aria, w),
            "baseline_vs_aria_histw":_histw(aria, b),
            "wearer_vs_aria_histw":  _histw(aria, w),
        }
        if ep.exists():
            e = _load(ep)
            row["egoworld_vs_aria_ssim"]  = _ssim(aria, e)
            row["egoworld_vs_aria_histw"] = _histw(aria, e)
        rows.append(row)
    out = ROOT / "eval_cam04"; out.mkdir(parents=True, exist_ok=True)
    (out/"paired_metrics.json").write_text(json.dumps(rows, indent=2))
    def summ(k):
        vs = [r[k] for r in rows if k in r]
        return f"{st.mean(vs):.3f} +/- {st.stdev(vs):.3f} (n={len(vs)})" if len(vs)>1 else "n<2"
    summary = {
        "n_paired": len(rows),
        "baseline_vs_aria_ssim":   summ("baseline_vs_aria_ssim"),
        "wearer_vs_aria_ssim":     summ("wearer_vs_aria_ssim"),
        "egoworld_vs_aria_ssim":   summ("egoworld_vs_aria_ssim"),
        "baseline_vs_aria_histw":  summ("baseline_vs_aria_histw"),
        "wearer_vs_aria_histw":    summ("wearer_vs_aria_histw"),
        "egoworld_vs_aria_histw":  summ("egoworld_vs_aria_histw"),
    }
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # 5-up comparison video
    T = (440, 320); hdr = 28
    frames_video = out/"comparison_frames"; frames_video.mkdir(parents=True, exist_ok=True)
    for r in rows:
        job = next(j for j in plan["jobs"] if j["idx"] == r["idx"])
        exo  = cv2.resize(cv2.imread(job["exo"]), T)
        b    = cv2.resize(cv2.imread(job["out_baseline"]), T)
        w    = cv2.resize(cv2.imread(job["out_wearer"]), T)
        aria = cv2.resize(cv2.imread(job["aria_gt"]), T)
        ego  = cv2.resize(cv2.imread(job["out_egoworld"]), T) if Path(job["out_egoworld"]).exists() else np.full((*T[::-1],3), 80, np.uint8)
        gap_v = np.full((T[1], 5, 3), 240, np.uint8)
        gap_h = np.full((hdr, 5*T[0] + 4*5, 3), 245, np.uint8)
        labels = ["cam04 exo", "baseline ego", "wearer-cond ego", "EgoWorld sparse-RGB inpaint", "Aria ego GT"]
        for i,L in enumerate(labels):
            cv2.putText(gap_h, L, (i*(T[0]+5)+8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20,20,20), 1)
        strip = np.concatenate([exo, gap_v, b, gap_v, w, gap_v, ego, gap_v, aria], axis=1)
        frame = np.concatenate([gap_h, strip], axis=0)
        cv2.imwrite(str(frames_video/f"frame_{r['idx']:03d}.jpg"), frame)
    video_out = ROOT/"clips"/"comparison_5up.mp4"; video_out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg","-y","-loglevel","error","-framerate","2",
        "-pattern_type","glob","-i", str(frames_video/"frame_*.jpg"),
        "-c:v","libx264","-pix_fmt","yuv420p","-r","30", str(video_out)
    ], check=False)
    print(f"wrote {video_out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("plan"); p1.add_argument("--n", type=int, default=10); p1.add_argument("--out", default=str(ROOT/"pipeline_plan.json"))
    p2 = sub.add_parser("generate"); p2.add_argument("--plan", default=str(ROOT/"pipeline_plan.json")); p2.add_argument("--skip-egoworld", action="store_true")
    p3 = sub.add_parser("eval"); p3.add_argument("--plan", default=str(ROOT/"pipeline_plan.json"))
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.cmd == "plan": cmd_plan(a.n, Path(a.out))
    elif a.cmd == "generate": cmd_generate(Path(a.plan), a.skip_egoworld)
    elif a.cmd == "eval": cmd_eval(Path(a.plan))


if __name__ == "__main__":
    main()
