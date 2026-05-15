#!/usr/bin/env python3
"""Full three-arm pipeline runner for the user's own paired capture.

Usage:
    python scripts/own_capture_pipeline.py plan --take take1 --n 10
    python scripts/own_capture_pipeline.py generate --take take1
    python scripts/own_capture_pipeline.py eval --take take1
"""
from __future__ import annotations
import argparse, base64, json, os, statistics as st, subprocess
from pathlib import Path
import cv2, numpy as np


def get_root(take: str) -> Path:
    return Path("data/own_capture") / take


def _frame_idx(path: str | Path) -> int | None:
    """Extract the trailing numeric frame id from names like exo_013.jpg."""
    stem = Path(path).stem
    try:
        return int(stem.split("_")[-1])
    except (ValueError, IndexError):
        return None


def _pairs_path(root: Path, pairs_file: str = "pairs.json") -> Path:
    p = Path(pairs_file)
    return p if p.is_absolute() else root / p


def _load_pairs(root: Path, pairs_file: str = "pairs.json") -> list[dict]:
    pairs_path = _pairs_path(root, pairs_file)
    if not pairs_path.exists():
        raise FileNotFoundError(f"missing synced pairs file: {pairs_path}")
    return json.loads(pairs_path.read_text())["pairs"]


def _load_pair_lookup(root: Path, pairs_file: str = "pairs.json") -> dict[int, dict]:
    """Map exo frame index -> synced pair record from pairs.json."""
    pairs = _load_pairs(root, pairs_file)
    return {int(p["exo_idx"]): p for p in pairs}


def cmd_plan(take: str, n: int, pairs_file: str = "pairs.json", plan_file: str = "pipeline_plan.json") -> None:
    root = get_root(take)
    pairs = _load_pairs(root, pairs_file)
    indices = [int(round(i*(len(pairs)-1)/max(n-1,1))) for i in range(n)]
    seen = []
    for i in indices:
        if i not in seen: seen.append(i)
    # Build base + wearer-conditioned prompts on-the-fly
    base_prompt = (root/"prompts"/"base.txt")
    base_prompt.parent.mkdir(parents=True, exist_ok=True)
    base_prompt.write_text(
        "Transform this third-person frame of a person sitting at a small round wooden table "
        "into a realistic first-person egocentric view from a phone-mounted-on-a-beanie camera "
        "worn by that person. The person is stacking books on the table. The egocentric camera "
        "is at the wearer's forehead, looking down at the books on the table. Preserve the same "
        "books (a dark hardcover and an orange-spined paperback titled 'The Beginning of Infinity'), "
        "the same hand positions, the same wooden table, and the same living-room background "
        "(white wall, window with blinds, gray couch). The wearer's own arms should enter the "
        "ego frame from the bottom. Mild fish-eye look. No extra hands, no extra people, no "
        "different furniture.\n"
    )

    wp = json.loads((root/"wearer_pose_exo"/"wearer_pose.json").read_text())
    by_frame = {Path(r["frame"]).stem: r for r in wp.get("rows",[]) if r.get("wearer_visible")}
    wp_dir = root/"wearer_conditioned_prompts"
    wp_dir.mkdir(parents=True, exist_ok=True)

    def qual_x(x):
        if x<0.25: return "the left side"
        if x<0.45: return "the left-of-center"
        if x<0.55: return "the center"
        if x<0.75: return "the right-of-center"
        return "the right side"
    def qual_y(y):
        if y<0.25: return "near the top"
        if y<0.45: return "the upper half"
        if y<0.55: return "vertical middle"
        if y<0.75: return "the lower half"
        return "near the bottom"

    base = base_prompt.read_text()
    for stem, row in by_frame.items():
        ec = row.get("implied_eye_norm") or [0.5,0.5]
        body = row.get("body_side") or "center"
        orient = row.get("head_orientation") or "unknown"
        block = (
            "\n\nWearer / ego-camera anchor (extracted from the exo frame):\n"
            f"- The person's head/eye region is at {qual_x(ec[0])} of the frame, {qual_y(ec[1])}.\n"
            f"- Body relative position in the frame: {body}.\n"
            f"- Head orientation: {orient}.\n"
            "- Therefore the egocentric viewpoint should be at the person's head, "
            "looking straight down at the books on the table, with their arms "
            "entering the ego frame from below.\n"
        )
        (wp_dir/f"prompt_{stem}.txt").write_text(base + block)

    jobs = []
    out_b = root/"generated_baseline"; out_b.mkdir(parents=True, exist_ok=True)
    out_w = root/"generated_wearer";   out_w.mkdir(parents=True, exist_ok=True)
    out_e = root/"generated_egoworld"; out_e.mkdir(parents=True, exist_ok=True)
    sparse_dir = root/"egoworld_sparse"; sparse_dir.mkdir(parents=True, exist_ok=True)
    for i in seen:
        p = pairs[i]
        exo = Path(p["exo"])
        ego = Path(p["ego"])
        wp_file = wp_dir/f"prompt_{exo.stem}.txt"
        if not wp_file.exists(): continue
        jobs.append({
            "idx": i,
            "exo": str(exo),
            "ego_gt": str(ego),
            "exo_idx_in_pairs": p["exo_idx"],
            "ego_idx_in_pairs": p["ego_idx"],
            "baseline_prompt": str(base_prompt),
            "wearer_prompt": str(wp_file),
            "out_baseline": str(out_b/f"{exo.stem}.png"),
            "out_wearer":   str(out_w/f"{exo.stem}.png"),
            "out_egoworld": str(out_e/f"{exo.stem}.png"),
            "sparse_dir":   str(sparse_dir/exo.stem),
        })
    out_path = root/plan_file
    out_path.write_text(json.dumps({
        "n_jobs": len(jobs),
        "pairs": str(_pairs_path(root, pairs_file)),
        "sync": {
            "source": pairs_file,
            "note": "ego_gt paths were resolved from synced pair records at plan time; eval re-validates them.",
        },
        "jobs": jobs,
    }, indent=2))
    print(f"wrote {out_path}  with {len(jobs)} jobs ({3*len(jobs)} API calls total)")


def _edit_call(client, model, image, prompt, size, mask=None):
    kw = dict(model=model, prompt=prompt, size=size, quality="medium")
    with open(image,"rb") as f:
        if mask:
            with open(mask,"rb") as fm:
                try: return client.images.edit(image=f, mask=fm, input_fidelity="high", **kw)
                except TypeError: pass
        else:
            try: return client.images.edit(image=f, input_fidelity="high", **kw)
            except TypeError: pass
    with open(image,"rb") as f:
        if mask:
            with open(mask,"rb") as fm:
                return client.images.edit(image=f, mask=fm, **kw)
        return client.images.edit(image=f, **kw)


def cmd_generate(take: str, skip_egoworld: bool=False, plan_file: str = "pipeline_plan.json") -> None:
    root = get_root(take)
    plan = json.loads((root/plan_file).read_text())
    try:
        from dotenv import load_dotenv; load_dotenv()
    except Exception: pass
    if not os.environ.get("OPENAI_API_KEY"): raise SystemExit("OPENAI_API_KEY not set.")
    from openai import OpenAI
    client = OpenAI()
    model = os.environ.get("EGOJUDGE_IMAGE_MODEL","gpt-image-1")

    if not skip_egoworld:
        for job in plan["jobs"]:
            sd = Path(job["sparse_dir"])
            if (sd/"hole_mask_rgba.png").exists(): continue
            print(f"build sparse for idx {job['idx']:03d}")
            subprocess.run(["python3","scripts/egoworld_pathway.py",
                            "--exo",job["exo"],"--out",str(sd),"--skip-inpaint"], check=False)

    for job in plan["jobs"]:
        for arm, key in [("baseline","baseline_prompt"),("wearer","wearer_prompt")]:
            out = Path(job[f"out_{arm}"])
            if out.exists():
                print(f"skip {out.name}"); continue
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                rsp = _edit_call(client, model, job["exo"], Path(job[key]).read_text(), "1536x1024")
                out.write_bytes(base64.b64decode(rsp.data[0].b64_json))
                print(f"  {arm}    idx {job['idx']:03d}")
            except Exception as e:
                print(f"  {arm}    idx {job['idx']:03d}  FAILED: {e}")
        if not skip_egoworld:
            out = Path(job["out_egoworld"])
            sparse_rgb = Path(job["sparse_dir"])/"sparse_ego_rgb.png"
            sparse_mask = Path(job["sparse_dir"])/"hole_mask_rgba.png"
            if out.exists() or not sparse_rgb.exists() or not sparse_mask.exists(): continue
            out.parent.mkdir(parents=True, exist_ok=True)
            prompt = (
                "This is a sparse egocentric (head-mounted, looking down at a tabletop) view "
                "reprojected from a third-person frame. Transparent pixels are missing and "
                "must be inpainted. Preserve every visible pixel. Fill the transparent regions "
                "so the result looks like a real GoPro/iPhone POV of a person stacking books on "
                "a small round wooden table: same books, same table, hands entering from below."
            )
            try:
                rsp = _edit_call(client, model, sparse_rgb, prompt, "1024x1024", mask=sparse_mask)
                out.write_bytes(base64.b64decode(rsp.data[0].b64_json))
                print(f"  egoworld idx {job['idx']:03d}")
            except Exception as e:
                print(f"  egoworld idx {job['idx']:03d}  FAILED: {e}")


def _load(p, size=(640,480)):
    im = cv2.resize(cv2.imread(str(p)), size)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _ssim(a,b):
    from scipy.ndimage import uniform_filter
    x = cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b,cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1,c2 = (0.01*255)**2, (0.03*255)**2
    ux,uy = uniform_filter(x,11), uniform_filter(y,11)
    vx = uniform_filter(x*x,11)-ux*ux; vy = uniform_filter(y*y,11)-uy*uy
    vxy = uniform_filter(x*y,11)-ux*uy
    return float(np.mean(((2*ux*uy+c1)*(2*vxy+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2)+1e-12)))


def _histw(a,b):
    from scipy.stats import wasserstein_distance
    bins = np.arange(256); ds=[]
    for ch in range(3):
        ha = cv2.calcHist([a],[ch],None,[256],[0,256]).ravel()
        hb = cv2.calcHist([b],[ch],None,[256],[0,256]).ravel()
        ha/=max(ha.sum(),1); hb/=max(hb.sum(),1)
        ds.append(wasserstein_distance(bins,bins,ha,hb))
    return float(np.mean(ds))


def cmd_eval(take: str, plan_file: str = "pipeline_plan.json", pairs_file: str | None = None) -> None:
    root = get_root(take)
    plan = json.loads((root/plan_file).read_text())
    pairs_file = pairs_file or plan.get("sync", {}).get("source") or plan.get("pairs") or "pairs.json"
    pair_lookup = _load_pair_lookup(root, pairs_file)
    rows = []
    skipped = []
    for job in plan["jobs"]:
        bp,wp,ep = map(Path, [job["out_baseline"],job["out_wearer"],job["out_egoworld"]])
        if not bp.exists() or not wp.exists():
            skipped.append({
                "idx": job.get("idx"),
                "exo": job.get("exo"),
                "reason": "missing baseline or wearer-generated image",
            })
            continue

        exo_idx = _frame_idx(job["exo"])
        pair = pair_lookup.get(exo_idx) if exo_idx is not None else None
        if pair is None:
            skipped.append({
                "idx": job.get("idx"),
                "exo": job.get("exo"),
                "reason": "exo frame is not present in current synced pairs.json",
            })
            continue

        plan_ego = Path(job.get("ego_gt", ""))
        pair_ego = Path(pair["ego"])
        if plan_ego != pair_ego:
            skipped.append({
                "idx": job.get("idx"),
                "exo": job.get("exo"),
                "reason": "stale plan ego_gt overridden by current pairs.json",
                "plan_ego_gt": str(plan_ego),
                "synced_ego_gt": str(pair_ego),
            })

        exo = _load(pair["exo"]); ego = _load(pair_ego)
        b = _load(bp); w = _load(wp)
        row = {
            "idx": job["idx"],
            "exo_idx": pair["exo_idx"],
            "ego_idx": pair["ego_idx"],
            "exo": pair["exo"],
            "ego_gt": pair["ego"],
            "baseline_vs_ego_ssim": _ssim(ego,b),
            "wearer_vs_ego_ssim":   _ssim(ego,w),
            "baseline_vs_ego_histw":_histw(ego,b),
            "wearer_vs_ego_histw":  _histw(ego,w),
        }
        if ep.exists():
            e = _load(ep)
            row["egoworld_vs_ego_ssim"]  = _ssim(ego,e)
            row["egoworld_vs_ego_histw"] = _histw(ego,e)
        rows.append(row)
    plan_stem = Path(plan_file).stem
    out_name = "eval" if plan_stem == "pipeline_plan" else f"eval_{plan_stem.replace('pipeline_plan_', '')}"
    out = root/out_name; out.mkdir(parents=True, exist_ok=True)
    (out/"paired_metrics.json").write_text(json.dumps({
        "rows": rows,
        "skipped_or_corrected": skipped,
        "sync_source": str(_pairs_path(root, pairs_file)),
        "plan_file": str(root / plan_file),
    }, indent=2))
    def summ(k):
        vs=[r[k] for r in rows if k in r]
        if len(vs) > 1:
            return f"{st.mean(vs):.3f} +/- {st.stdev(vs):.3f} (n={len(vs)})"
        if len(vs) == 1:
            return f"{vs[0]:.3f} (n=1)"
        return "n=0"
    summary = {
        "n_paired": len(rows),
        "n_skipped_or_corrected": len(skipped),
        "sync_source": str(_pairs_path(root, pairs_file)),
        "baseline_vs_ego_ssim":   summ("baseline_vs_ego_ssim"),
        "wearer_vs_ego_ssim":     summ("wearer_vs_ego_ssim"),
        "egoworld_vs_ego_ssim":   summ("egoworld_vs_ego_ssim"),
        "baseline_vs_ego_histw":  summ("baseline_vs_ego_histw"),
        "wearer_vs_ego_histw":    summ("wearer_vs_ego_histw"),
        "egoworld_vs_ego_histw":  summ("egoworld_vs_ego_histw"),
    }
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # 5-up comparison video
    T=(440,320); hdr=28
    fv = out/"comparison_frames"; fv.mkdir(parents=True, exist_ok=True)
    for r in rows:
        job = next(j for j in plan["jobs"] if j["idx"]==r["idx"])
        exo = cv2.resize(cv2.imread(r["exo"]), T)
        b   = cv2.resize(cv2.imread(job["out_baseline"]), T)
        w   = cv2.resize(cv2.imread(job["out_wearer"]), T)
        ego = cv2.resize(cv2.imread(r["ego_gt"]), T)
        ego_img = ego  # already rotated correctly in extracted frames
        eo  = Path(job["out_egoworld"])
        e = cv2.resize(cv2.imread(str(eo)), T) if eo.exists() else np.full((T[1],T[0],3),80,np.uint8)
        gap_v = np.full((T[1],5,3),240,np.uint8)
        gap_h = np.full((hdr,5*T[0]+4*5,3),245,np.uint8)
        labels = ["exo (ZV-E10)","baseline ego","wearer-cond ego","EgoWorld inpaint","ego GT (iPhone)"]
        for i,L in enumerate(labels):
            cv2.putText(gap_h, L, (i*(T[0]+5)+8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20,20,20), 1)
        strip = np.concatenate([exo,gap_v,b,gap_v,w,gap_v,e,gap_v,ego_img], axis=1)
        frame = np.concatenate([gap_h, strip], axis=0)
        cv2.imwrite(str(fv/f"frame_{r['idx']:03d}.jpg"), frame)
    video_suffix = "" if out_name == "eval" else f"_{out_name.removeprefix('eval_')}"
    video = root/"clips"/f"comparison_5up{video_suffix}.mp4"; video.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate","2",
                    "-pattern_type","glob","-i",str(fv/"frame_*.jpg"),
                    "-c:v","libx264","-pix_fmt","yuv420p","-r","30", str(video)], check=False)
    print(f"wrote {video}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("plan"); p1.add_argument("--take", default="take1"); p1.add_argument("--n", type=int, default=10); p1.add_argument("--pairs-file", default="pairs.json"); p1.add_argument("--plan-file", default="pipeline_plan.json")
    p2 = sub.add_parser("generate"); p2.add_argument("--take", default="take1"); p2.add_argument("--skip-egoworld", action="store_true"); p2.add_argument("--plan-file", default="pipeline_plan.json")
    p3 = sub.add_parser("eval"); p3.add_argument("--take", default="take1"); p3.add_argument("--plan-file", default="pipeline_plan.json"); p3.add_argument("--pairs-file", default=None)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.cmd == "plan": cmd_plan(a.take, a.n, a.pairs_file, a.plan_file)
    elif a.cmd == "generate": cmd_generate(a.take, a.skip_egoworld, a.plan_file)
    elif a.cmd == "eval": cmd_eval(a.take, a.plan_file, a.pairs_file)


if __name__ == "__main__":
    main()
