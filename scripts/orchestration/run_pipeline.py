"""End-to-end one-frame pipeline runner: exo → ego with full eval.

This is the single CLI the user asked for: drop in an exo frame, get all
artifacts (detections, scene graph, prompt, 3D viz, generated ego if
requested, constraint judge result, pipeline-step panel) in one folder.

It composes scripts that already exist:
  - scripts/detect_hosted_hf.py     Grounding-DINO detections + scene graph
  - scripts/viz_3d_reprojection.py  interactive 3D point cloud HTML
  - scripts/egojudge_constraints.py LLM constraint judge over the ego output
  - scripts/make_pipeline_panel.py  per-frame pipeline-step visualization

Each step writes its outputs into <out>/<step>/ and emits a manifest.json with
file paths + summary numbers. Skips steps whose outputs already exist (use
--force to recompute).

Usage:
    python scripts/run_pipeline.py \
        --exo experiments/new_upload_sync/take2/frames_full/exo/exo_009_130.324.jpg \
        --ego experiments/new_upload_sync/take2/generated/gen_009_exo_130.324.png \
        --prompt experiments/new_upload_sync/take2/variant_prompts/action_phase_strict.txt \
        --out experiments/new_upload_sync/take2/runs/exo_009 \
        --skip-judge   # omit if you want the OpenAI judge to run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import time


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent  # scripts/ root, contains perception/, eval/, viz/, ...


def run(cmd: list[str], dry: bool = False) -> int:
    print(">>", " ".join(cmd))
    if dry:
        return 0
    p = subprocess.run(cmd)
    return p.returncode


def step_detect(exo: Path, out: Path, force: bool) -> dict:
    target = out / "detect"
    overlay = target / "overlay.png"
    sg = target / "scene_graph.json"
    if overlay.exists() and sg.exists() and not force:
        print(f"[skip] detect: {overlay} exists")
    else:
        rc = run([sys.executable, str(SCRIPTS / "perception" / "detect_hosted_hf.py"),
                  "--image", str(exo), "--out", str(target),
                  "--labels", "hand. book. table. couch. window. tv. person. jar. salt. ghee.",
                  "--threshold", "0.30"])
        if rc != 0:
            return {"status": "fail", "step": "detect"}
    return {"status": "ok", "step": "detect",
            "scene_graph": str(sg), "overlay": str(overlay)}


def step_viz3d(exo: Path, out: Path, force: bool) -> dict:
    target = out / "viz_3d"
    html = target / "pointcloud_3d.html"
    if html.exists() and not force:
        print(f"[skip] viz_3d: {html} exists")
    else:
        rc = run([sys.executable, str(SCRIPTS / "viz" / "viz_3d_reprojection.py"),
                  "--exo", str(exo), "--out", str(target)])
        if rc != 0:
            return {"status": "fail", "step": "viz_3d"}
    return {"status": "ok", "step": "viz_3d", "html": str(html)}


def step_judge(exo: Path, ego: Path | None, prompt: Path | None,
               scene_graph: Path, out: Path, force: bool, model: str) -> dict:
    if ego is None or not ego.exists():
        return {"status": "skip", "step": "judge", "reason": "no ego candidate"}
    target = out / "judge"
    target.mkdir(parents=True, exist_ok=True)
    judge_json = target / "constraints.json"
    if judge_json.exists() and not force:
        print(f"[skip] judge: {judge_json} exists")
    else:
        cmd = [sys.executable, str(SCRIPTS / "eval" / "egojudge_constraints.py"),
               "--exo", str(exo), "--ego", str(ego),
               "--out", str(judge_json), "--model", model]
        if prompt is not None and prompt.exists():
            cmd += ["--prompt", str(prompt)]
        if scene_graph and scene_graph.exists():
            cmd += ["--scene-graph", str(scene_graph)]
        rc = run(cmd)
        if rc != 0:
            return {"status": "fail", "step": "judge"}
    return {"status": "ok", "step": "judge", "result": str(judge_json)}


def step_panel(exo: Path, ego: Path | None, out: Path, force: bool) -> dict:
    """Compose a per-frame pipeline-step panel: source / detect / 3D / ego."""
    from PIL import Image, ImageDraw, ImageFont
    target = out / "panel"
    target.mkdir(parents=True, exist_ok=True)
    panel_path = target / "pipeline_steps.png"
    if panel_path.exists() and not force:
        print(f"[skip] panel: {panel_path} exists")
        return {"status": "ok", "step": "panel", "panel": str(panel_path)}

    tiles = [
        ("1. Source exo (input)",       exo),
        ("2. Open-vocab detection",     out / "detect" / "overlay.png"),
        ("3. Depth + 3D point cloud",   out / "viz_3d" / "pointcloud_3d.png"),
        ("4. Generated ego candidate",  ego if ego else None),
    ]
    cell_w, cell_h, label_h, pad = 720, 405, 36, 14
    cols, rows = 2, 2
    W = cols * cell_w + (cols + 1) * pad
    H = rows * (cell_h + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for idx, (title, p) in enumerate(tiles):
        r, c = divmod(idx, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + label_h + pad)
        if p and Path(p).exists():
            img = Image.open(p).convert("RGB")
            img.thumbnail((cell_w, cell_h), Image.LANCZOS)
            cb = Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
            cb.paste(img, ((cell_w - img.size[0]) // 2, (cell_h - img.size[1]) // 2))
        else:
            cb = Image.new("RGB", (cell_w, cell_h), (40, 40, 40))
            d = ImageDraw.Draw(cb)
            d.text((20, 20), f"missing: {Path(str(p)).name if p else 'n/a'}",
                   fill=(220, 80, 80))
        canvas.paste(cb, (x, y))
        draw.text((x + 6, y + cell_h + 4), title, fill=(20, 20, 20), font=font)

    canvas.save(panel_path, optimize=True)
    return {"status": "ok", "step": "panel", "panel": str(panel_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exo", required=True)
    ap.add_argument("--ego", default=None, help="existing generated ego candidate to evaluate")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-detect", action="store_true")
    ap.add_argument("--skip-viz3d", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--skip-panel", action="store_true")
    ap.add_argument("--judge-model", default="gpt-5.1")
    args = ap.parse_args()

    exo = Path(args.exo)
    ego = Path(args.ego) if args.ego else None
    prompt = Path(args.prompt) if args.prompt else None
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if not exo.exists():
        raise SystemExit(f"exo not found: {exo}")

    manifest = {
        "exo": str(exo),
        "ego": str(ego) if ego else None,
        "prompt": str(prompt) if prompt else None,
        "out_dir": str(out),
        "started_at": int(time()),
        "steps": [],
    }

    if not args.skip_detect:
        manifest["steps"].append(step_detect(exo, out, args.force))
    if not args.skip_viz3d:
        manifest["steps"].append(step_viz3d(exo, out, args.force))
    sg = out / "detect" / "scene_graph.json"
    if not args.skip_judge:
        manifest["steps"].append(step_judge(exo, ego, prompt, sg, out, args.force,
                                            args.judge_model))
    if not args.skip_panel:
        manifest["steps"].append(step_panel(exo, ego, out, args.force))

    manifest["finished_at"] = int(time())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out / 'manifest.json'}")
    for s in manifest["steps"]:
        print(f"  {s.get('step'):<10} {s.get('status')}")


if __name__ == "__main__":
    main()
