"""Compute the 5-axis eval suite over a list of (exo, ego, gt, judge_json) tuples
and emit a Markdown + LaTeX table.

Usage: python scripts/eval/multi_frame_eval_rollup.py
       (paths are hard-coded for take2 + take3 frames — quick rollup for the report).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def paired(ego, gt):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    a = np.array(Image.open(ego).convert("RGB"))
    b = np.array(Image.open(gt).convert("RGB").resize(Image.open(ego).size))
    psnr = float(peak_signal_noise_ratio(b, a, data_range=255))
    g_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY); g_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    ssim = float(structural_similarity(g_a, g_b, data_range=255))
    e_a = cv2.Canny(g_a, 80, 200) > 0; e_b = cv2.Canny(g_b, 80, 200) > 0
    tp = float((e_a & e_b).sum()); fp = float((e_a & ~e_b).sum()); fn = float((~e_a & e_b).sum())
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return {"PSNR": psnr, "SSIM": ssim, "edge_F1": f1}


def depth_pair(ego, gt, _cache={}):
    from transformers import pipeline as hf_pipeline
    if "p" not in _cache:
        _cache["p"] = hf_pipeline(task="depth-estimation",
                                  model="depth-anything/Depth-Anything-V2-Small-hf")
    res = {}
    for tag, path in [("gen", ego), ("gt", gt)]:
        d = np.array(_cache["p"](Image.open(path).convert("RGB"))["depth"]).astype(np.float32)
        res[tag] = {"median": float(np.median(d)), "std": float(d.std())}
    res["abs_median_delta"] = abs(res["gen"]["median"] - res["gt"]["median"])
    return res


def robot(ego):
    try:
        import mediapipe as mp
    except Exception:
        return {"hand_count": None}
    rgb = np.array(Image.open(ego).convert("RGB"))
    h = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=4,
                                 model_complexity=1, min_detection_confidence=0.2)
    res = h.process(rgb)
    n = len(res.multi_hand_landmarks or [])
    return {"hand_count": n}


def view_overlap(exo, ego):
    a = cv2.imread(exo); b = cv2.imread(ego)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY); gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(ga, None); kp2, des2 = orb.detectAndCompute(gb, None)
    if des1 is None or des2 is None: return {"inliers": 0}
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2),
                     key=lambda m: m.distance)[:200]
    if len(matches) < 8: return {"inliers": len(matches)}
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return {"inliers": int(mask.sum()) if mask is not None else 0}


def judge(jpath):
    s = json.loads(Path(jpath).read_text())["judge_result"]["summary"]
    return {"pass": s["n_pass"], "fail": s["n_fail"], "score": s["score_weighted"],
            "verdict": s["verdict"]}


def main():
    FRAMES = [
        ("Take 2 exo_007", "experiments/new_upload_sync/take2/frames_full/exo/exo_007_126.280.jpg",
         "experiments/new_upload_sync/take2/generated/gen_007_exo_126.280.png",
         "experiments/new_upload_sync/take2/frames_full/ego_flipped/ego_flip_007_412.668.jpg",
         "experiments/new_upload_sync/take2/runs/exo_007/judge/constraints.json"),
        ("Take 2 exo_009", "experiments/new_upload_sync/take2/frames_full/exo/exo_009_130.324.jpg",
         "experiments/new_upload_sync/take2/generated/gen_009_exo_130.324.png",
         "experiments/new_upload_sync/take2/frames_full/ego_flipped/ego_flip_009_416.712.jpg",
         "experiments/new_upload_sync/take2/runs/exo_009/judge/constraints.json"),
        ("Take 2 exo_011", "experiments/new_upload_sync/take2/frames_full/exo/exo_011_134.521.jpg",
         "experiments/new_upload_sync/take2/generated/gen_011_exo_134.521.png",
         "experiments/new_upload_sync/take2/frames_full/ego_flipped/ego_flip_011_420.909.jpg",
         "experiments/new_upload_sync/take2/runs/exo_011/judge/constraints.json"),
        ("Take 3 exo_005", "experiments/new_upload_sync/take3/frames/exo/exo_005_145.325.jpg",
         "experiments/new_upload_sync/take3/generated/gen_005.png",
         "experiments/new_upload_sync/take3/frames/ego/ego_005_249.800.jpg",
         "experiments/new_upload_sync/take3/runs/exo_005/judge/constraints.json"),
        ("Take 3 exo_007", "experiments/new_upload_sync/take3/frames/exo/exo_007_153.325.jpg",
         "experiments/new_upload_sync/take3/generated/gen_007.png",
         "experiments/new_upload_sync/take3/frames/ego/ego_007_257.800.jpg",
         "experiments/new_upload_sync/take3/runs/exo_007/judge/constraints.json"),
        ("Take 3 exo_011", "experiments/new_upload_sync/take3/frames/exo/exo_011_169.325.jpg",
         "experiments/new_upload_sync/take3/generated/gen_011.png",
         "experiments/new_upload_sync/take3/frames/ego/ego_011_273.800.jpg",
         "experiments/new_upload_sync/take3/runs/exo_011/judge/constraints.json"),
    ]

    # filter by existence of paths
    valid = []
    for name, exo, ego, gt, jd in FRAMES:
        if all(Path(p).exists() for p in [exo, ego, gt, jd]):
            valid.append((name, exo, ego, gt, jd))
        else:
            print(f"[skip] {name}  (missing path)")

    rows = []
    for name, exo, ego, gt, jd in valid:
        print("==", name)
        rows.append({
            "frame": name,
            "paired": paired(ego, gt),
            "depth":  depth_pair(ego, gt),
            "robot":  robot(ego),
            "view":   view_overlap(exo, ego),
            "judge":  judge(jd),
        })

    out_md  = Path("experiments/new_upload_sync/eval_rollup.md")
    out_tex = Path("paper/eval_rollup.tex")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    # Markdown
    md = ["# Five-axis rollup across own captures (Take 2 + Take 3)\n",
          "| frame | PSNR | SSIM | edge F1 | depth |delta| med | hand count | view inliers | judge pass/fail | judge score | verdict |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        p = r["paired"]; d = r["depth"]; rb = r["robot"]; v = r["view"]; j = r["judge"]
        md.append(f"| {r['frame']} | {p['PSNR']:.2f} | {p['SSIM']:.2f} | {p['edge_F1']:.3f} | "
                  f"{d['abs_median_delta']:.0f} | {rb['hand_count']} | {v['inliers']} | "
                  f"{j['pass']}/{j['fail']} | {j['score']:.2f} | {j['verdict']} |")
    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md}")

    # LaTeX (booktabs)
    bsl = "\\\\"
    tex = [r"\begin{table}[H]\centering\small",
           r"\caption{Five-axis eval rollup across the six paired own-capture frames. "
           r"Axis~1 paired metrics on the generated ego vs.\ the 180$^\circ$-corrected ego GT. "
           r"Axis~2 hand count from MediaPipe Hands on the generated ego. "
           r"Axis~3 depth-median delta between Depth Anything V2 outputs of gen and GT. "
           r"Axis~4 ORB+RANSAC homography inliers from gen back to exo. "
           r"Axis~5 LLM constraint judge pass/fail count, weighted score, verdict.}",
           r"\label{tab:eval-rollup}",
           r"\begin{tabular}{l rrr r r r r r l}",
           r"\toprule",
           r" & \multicolumn{3}{c}{Paired} & Depth & Hand & View & \multicolumn{2}{c}{Judge} & \\",
           r"\cmidrule(lr){2-4} \cmidrule(lr){8-9}",
           r"Frame & PSNR & SSIM & edge F1 & $|\Delta$med$|$ & count & inliers & pass/fail & score & verdict " + bsl,
           r"\midrule"]
    for r in rows:
        p, d, rb, v, j = r["paired"], r["depth"], r["robot"], r["view"], r["judge"]
        tex.append(f"{r['frame']} & {p['PSNR']:.2f} & {p['SSIM']:.2f} & {p['edge_F1']:.3f} & "
                   f"{d['abs_median_delta']:.0f} & {rb['hand_count']} & {v['inliers']} & "
                   f"{j['pass']}/{j['fail']} & {j['score']:.2f} & {j['verdict']} {bsl}")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_tex.write_text("\n".join(tex) + "\n")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
