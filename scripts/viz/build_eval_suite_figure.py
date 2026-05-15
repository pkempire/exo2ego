"""Single-figure visualization of the FULL 5-axis EgoJudge eval suite.

Not just the LLM judge: paired pixel metrics, depth-physics agreement,
robot-usefulness (hand count + contact), ORB+RANSAC view overlap, and the
constraint judge. All computed on the same Take 2 exo_009 frame.

Usage:
  python scripts/viz/build_eval_suite_figure.py \
    --exo experiments/new_upload_sync/take2/frames_full/exo/exo_009_130.324.jpg \
    --ego experiments/new_upload_sync/take2/generated/gen_009_exo_130.324.png \
    --gt  experiments/new_upload_sync/take2/frames_full/ego_flipped/ego_flip_009_385.762.jpg \
    --judge experiments/new_upload_sync/take2/runs/exo_009/judge/constraints.json \
    --out paper/figures/eval_suite_in_action.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------- per-axis computations ----------------

def axis_paired(ego_path: str, gt_path: str) -> dict:
    """PSNR, SSIM, edge F1 between generated ego and (180°-corrected) GT ego."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    a = np.array(Image.open(ego_path).convert("RGB"))
    b = np.array(Image.open(gt_path).convert("RGB").resize(Image.open(ego_path).size))
    psnr = float(peak_signal_noise_ratio(b, a, data_range=255))
    g_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    g_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    ssim = float(structural_similarity(g_a, g_b, data_range=255))
    e_a = cv2.Canny(g_a, 80, 200) > 0
    e_b = cv2.Canny(g_b, 80, 200) > 0
    tp = float((e_a & e_b).sum())
    fp = float((e_a & ~e_b).sum())
    fn = float((~e_a & e_b).sum())
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return {"PSNR": psnr, "SSIM": ssim, "edge_F1": f1}


def axis_depth(ego_path: str, gt_path: str, cache_dir: Path | None = None):
    """Depth Anything V2 on both; return mean/std/median of each + a side-by-side viz."""
    from transformers import pipeline as hf_pipeline
    import matplotlib.cm as cm
    pipe = hf_pipeline(task="depth-estimation",
                       model="depth-anything/Depth-Anything-V2-Small-hf")

    def colorize(arr):
        a = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return Image.fromarray((cm.viridis(a)[:, :, :3] * 255).astype(np.uint8))

    res = {}
    panels = {}
    for tag, p in [("gen", ego_path), ("gt", gt_path)]:
        rgb = Image.open(p).convert("RGB")
        out = pipe(rgb)
        d = np.array(out["depth"]).astype(np.float32)
        res[tag] = {"mean": float(d.mean()), "std": float(d.std()),
                    "median": float(np.median(d))}
        panels[tag] = colorize(d)
    res["delta_mean"] = res["gen"]["mean"] - res["gt"]["mean"]
    res["delta_median"] = res["gen"]["median"] - res["gt"]["median"]
    return res, panels


def axis_robot(ego_path: str) -> dict:
    """Hand count + hand_low fraction via MediaPipe Hands."""
    try:
        import mediapipe as mp
    except Exception:
        return {"hand_count": None, "hand_in_lower_half": None, "note": "mediapipe missing"}
    rgb = np.array(Image.open(ego_path).convert("RGB"))
    h_model = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=4,
                                       model_complexity=1, min_detection_confidence=0.2)
    res = h_model.process(rgb)
    n = len(res.multi_hand_landmarks or [])
    lowery = []
    if res.multi_hand_landmarks:
        for h in res.multi_hand_landmarks:
            ys = [lm.y for lm in h.landmark]
            lowery.append(np.mean(ys))
    in_lower = sum(1 for y in lowery if y > 0.55)
    return {"hand_count": n, "hands_in_lower_half": in_lower,
            "median_wrist_y": float(np.median(lowery)) if lowery else None}


def axis_view_overlap(exo_path: str, ego_path: str):
    """ORB + RANSAC homography from generated ego to source exo; returns inliers + polygon."""
    a = cv2.imread(exo_path); b = cv2.imread(ego_path)
    if a is None or b is None:
        return {"inliers": 0}, None
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY); gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(ga, None)
    kp2, des2 = orb.detectAndCompute(gb, None)
    if des1 is None or des2 is None:
        return {"inliers": 0}, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)[:200]
    if len(matches) < 8:
        return {"inliers": len(matches)}, None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return {"inliers": inliers, "n_matches": len(matches)}, (a, b, H, kp1, kp2, matches, mask)


def axis_judge(judge_path: str) -> dict:
    d = json.loads(Path(judge_path).read_text())
    s = d["judge_result"]["summary"]
    fails = (s.get("top_failures") or [])[:3]
    return {"pass": s.get("n_pass"), "fail": s.get("n_fail"),
            "uncertain": s.get("n_uncertain"),
            "score": s.get("score_weighted"), "verdict": s.get("verdict"),
            "top_failures": fails}


# ---------------- layout ----------------

def render_text_panel(title: str, lines: list[str], size, color_bg=(245, 247, 252),
                      color_title=(60, 90, 160)):
    img = Image.new("RGB", size, color_bg)
    draw = ImageDraw.Draw(img)
    try:
        tfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        bfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        tfont = bfont = ImageFont.load_default()
    draw.rectangle([0, 0, size[0], 32], fill=color_title)
    draw.text((10, 6), title, fill=(255, 255, 255), font=tfont)
    y = 42
    for ln in lines:
        if y > size[1] - 22:
            break
        draw.text((10, y), ln, fill=(20, 20, 20), font=bfont)
        y += 22
    return img


def paste(canvas, img, x, y, size, label, font):
    im = img.copy(); im.thumbnail(size, Image.LANCZOS)
    cb = Image.new("RGB", size, (0, 0, 0))
    cb.paste(im, ((size[0] - im.size[0]) // 2, (size[1] - im.size[1]) // 2))
    canvas.paste(cb, (x, y))
    d = ImageDraw.Draw(canvas)
    d.text((x + 6, y + size[1] + 4), label, fill=(20, 20, 20), font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exo", required=True)
    ap.add_argument("--ego", required=True, help="generated ego candidate")
    ap.add_argument("--gt",  required=True, help="180°-corrected GT ego frame")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("computing paired ..."); paired = axis_paired(args.ego, args.gt)
    print("computing depth ...");  depth_res, depth_panels = axis_depth(args.ego, args.gt)
    print("computing robot ...");  robot = axis_robot(args.ego)
    print("computing view ...");   view_res, view_meta = axis_view_overlap(args.exo, args.ego)
    print("loading judge ...");    judge = axis_judge(args.judge)

    # Build view-overlap viz: draw exo-plane polygon projected into ego
    if view_meta is not None:
        a, b, H, kp1, kp2, matches, mask = view_meta
        viz_b = b.copy()
        if H is not None:
            h, w = a.shape[:2]
            corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(corners, H)
            cv2.polylines(viz_b, [np.int32(proj)], True, (255, 0, 255), 6)
        viz_b_pil = Image.fromarray(cv2.cvtColor(viz_b, cv2.COLOR_BGR2RGB))
    else:
        viz_b_pil = Image.open(args.ego).convert("RGB")

    # ----- canvas layout -----
    # Top row: exo | gen | gt   (3 cells)
    # Mid row: depth_gen | depth_gt | view_overlap (3 cells)
    # Bottom row: paired text | robot text | judge text  (3 cells)
    cell_w, cell_h, label_h, pad = 540, 304, 32, 14
    top_title_h = 50
    cols, rows = 3, 3
    W = cols * cell_w + (cols + 1) * pad
    H = top_title_h + rows * (cell_h + label_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        tf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        lf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        tf = lf = ImageFont.load_default()

    from pathlib import Path as _P
    frame_label = _P(args.exo).stem
    draw.text((pad+6, 12),
              f"EgoJudge eval suite -- all five axes on one frame  ({frame_label})",
              fill=(20, 20, 20), font=tf)

    def cell_xy(r, c):
        return (pad + c * (cell_w + pad),
                top_title_h + pad + r * (cell_h + label_h + pad))

    # Row 0: source/gen/gt
    paste(canvas, Image.open(args.exo).convert("RGB"), *cell_xy(0, 0), (cell_w, cell_h),
          "source exo", lf)
    paste(canvas, Image.open(args.ego).convert("RGB"), *cell_xy(0, 1), (cell_w, cell_h),
          "generated ego", lf)
    paste(canvas, Image.open(args.gt).convert("RGB"), *cell_xy(0, 2), (cell_w, cell_h),
          "paired ego GT", lf)

    # Row 1: depth gen / depth gt / view overlap viz
    paste(canvas, depth_panels["gen"], *cell_xy(1, 0), (cell_w, cell_h),
          f"axis 3 -- Depth Anything V2 (gen)  median={depth_res['gen']['median']:.0f}", lf)
    paste(canvas, depth_panels["gt"],  *cell_xy(1, 1), (cell_w, cell_h),
          f"axis 3 -- Depth Anything V2 (GT)   median={depth_res['gt']['median']:.0f}", lf)
    paste(canvas, viz_b_pil,           *cell_xy(1, 2), (cell_w, cell_h),
          f"axis 4 -- ORB+RANSAC view overlap   inliers={view_res['inliers']}", lf)

    # Row 2: paired text / robot text / judge text
    paired_panel = render_text_panel(
        "axis 1 -- Paired pixel similarity",
        [
            f"PSNR        {paired['PSNR']:.2f}",
            f"SSIM        {paired['SSIM']:.3f}",
            f"edge F1     {paired['edge_F1']:.3f}",
            "",
            "(only available when GT ego is paired;",
            " low values are expected on hand-shot ego)",
        ],
        (cell_w, cell_h),
        color_title=(31, 119, 180),
    )
    canvas.paste(paired_panel, cell_xy(2, 0))
    draw.text((cell_xy(2,0)[0]+6, cell_xy(2,0)[1]+cell_h+4),
              "scripts/eval/egojudge_metrics.py", fill=(20,20,20), font=lf)

    robot_panel = render_text_panel(
        "axis 2 -- Robot usefulness (MediaPipe)",
        [
            f"hand_count               {robot.get('hand_count')}",
            f"hands in lower half      {robot.get('hands_in_lower_half')}",
            (f"median wrist y           {robot.get('median_wrist_y'):.2f}"
             if robot.get('median_wrist_y') is not None else "median wrist y           --"),
            "",
            "(catches extra hands, missing hands,",
            " hands not entering from below)",
        ],
        (cell_w, cell_h),
        color_title=(44, 160, 44),
    )
    canvas.paste(robot_panel, cell_xy(2, 1))
    draw.text((cell_xy(2,1)[0]+6, cell_xy(2,1)[1]+cell_h+4),
              "scripts/eval/robot_usefulness_score.py", fill=(20,20,20), font=lf)

    n_total = judge['pass'] + judge['fail'] + judge['uncertain']
    judge_lines = [
        f"pass / fail / uncertain    {judge['pass']} / {judge['fail']} / {judge['uncertain']}",
        f"weighted pass fraction     {judge['score']:.2f}",
        f"  (always-on fails 2x)",
        f"  (n_total constraints     {n_total})",
        "",
        "top failures:",
    ] + [f"  - {f[:50]}" for f in judge["top_failures"]]
    judge_panel = render_text_panel(
        "axis 5 -- LLM constraint judge (gpt-5.1)",
        judge_lines,
        (cell_w, cell_h),
        color_title=(111, 66, 193),
    )
    canvas.paste(judge_panel, cell_xy(2, 2))
    draw.text((cell_xy(2,2)[0]+6, cell_xy(2,2)[1]+cell_h+4),
              "scripts/eval/egojudge_constraints.py", fill=(20,20,20), font=lf)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(f"wrote {args.out}  {canvas.size}")


if __name__ == "__main__":
    main()
