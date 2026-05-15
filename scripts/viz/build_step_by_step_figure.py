"""One-figure walkthrough of the EgoJudge pipeline on a single frame.

Eight panels, 2 rows x 4 cols:
  1. Source exo
  2. Grounding-DINO open-vocab boxes
  3. MediaPipe hand landmarks (21 keypoints per hand) overlaid on exo
  4. Depth Anything V2 colorized depth map
  5. Auto-built scene graph bbox layout (hands / workspace / manipulated objects)
  6. Auto-built generation prompt (rendered card showing the scene-graph JSON
     embedded in the prompt -- this proves the prompt is perception-conditioned
     with real normalized coordinates, not a hand-written caption)
  7. Generated ego candidate (gpt-image-1 edit on exo + auto prompt)
  8. Paired GT ego (when available; otherwise blank)

Usage:
  python scripts/viz/build_step_by_step_figure.py \
    --exo experiments/new_upload_sync/take2/frames_full/exo/exo_009_130.324.jpg \
    --dino experiments/new_upload_sync/take2/runs/exo_009/detect/overlay.png \
    --scene-graph experiments/new_upload_sync/take2/runs/exo_009/detect/scene_graph.json \
    --auto-prompt experiments/new_upload_sync/take3/auto_prompt_007/prompt_conditioned_vlm.txt \
    --ego experiments/new_upload_sync/take2/generated/gen_009_exo_130.324.png \
    --gt experiments/new_upload_sync/take2/frames_full/ego_flipped/ego_flip_009_416.712.jpg \
    --out paper/figures/pipeline_step_by_step_take2.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def colorize_depth(arr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    a = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return Image.fromarray((cm.viridis(a)[:, :, :3] * 255).astype(np.uint8))


def get_depth(exo_path: str, cache_path: str | None) -> Image.Image:
    if cache_path and Path(cache_path).exists():
        return Image.open(cache_path)
    from transformers import pipeline as hf_pipeline
    rgb = Image.open(exo_path).convert("RGB")
    pipe = hf_pipeline(task="depth-estimation",
                       model="depth-anything/Depth-Anything-V2-Small-hf")
    out = pipe(rgb)
    arr = np.array(out["depth"]).astype(np.float32)
    color = colorize_depth(arr)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        color.save(cache_path)
    return color


def render_hand_landmarks(exo_path: str) -> Image.Image:
    """Overlay MediaPipe hand landmarks (21 per hand) on the exo frame."""
    try:
        import mediapipe as mp
    except Exception:
        img = Image.open(exo_path).convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "mediapipe missing", fill=(255, 0, 0), font=load_font(28))
        return img
    rgb = np.array(Image.open(exo_path).convert("RGB"))
    H, W, _ = rgb.shape
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                     model_complexity=1, min_detection_confidence=0.2)
    res = hands.process(rgb)
    img = Image.fromarray(rgb).copy()
    draw = ImageDraw.Draw(img, "RGBA")
    conns = mp.solutions.hands.HAND_CONNECTIONS
    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            pts = [(int(lm.x * W), int(lm.y * H)) for lm in hand.landmark]
            for a, b in conns:
                draw.line([pts[a], pts[b]], fill=(50, 220, 90, 255), width=4)
            for (x, y) in pts:
                draw.ellipse([x-5, y-5, x+5, y+5], fill=(255, 80, 80, 255),
                             outline=(255, 255, 255, 255), width=1)
    else:
        draw.text((20, 20), "no hands detected", fill=(255, 80, 80), font=load_font(28))
    return img


def render_scene_graph_overlay(exo_path: str, sg_path: str) -> Image.Image:
    sg = json.loads(Path(sg_path).read_text())
    img = Image.open(exo_path).convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font = load_font(20)
    ws = sg.get("workspace") or {}
    b = ws.get("bbox_norm")
    if b:
        x, y, w, h = b
        draw.rectangle([int(x*W), int(y*H), int((x+w)*W), int((y+h)*H)],
                       outline=(255, 200, 80), width=5)
        draw.text((int(x*W)+6, int(y*H)+4), "workspace", fill=(255, 200, 80), font=font)
    for hand in sg.get("hands") or []:
        b = hand.get("bbox_norm")
        if not b: continue
        x, y, w, h = b
        draw.rectangle([int(x*W), int(y*H), int((x+w)*W), int((y+h)*H)],
                       outline=(255, 90, 90), width=5)
        draw.text((int(x*W)+6, int(y*H)+4), hand.get("name", "hand"),
                  fill=(255, 90, 90), font=font)
    for obj in sg.get("objects") or []:
        b = obj.get("bbox_norm")
        if not b: continue
        x, y, w, h = b
        draw.rectangle([int(x*W), int(y*H), int((x+w)*W), int((y+h)*H)],
                       outline=(90, 200, 90), width=5)
        draw.text((int(x*W)+6, int(y*H)+4), obj.get("name", ""),
                  fill=(90, 200, 90), font=font)
    return img


def render_prompt_card(auto_prompt_path: str, size) -> Image.Image:
    """Render the prompt as a coloured card showing both the natural-language wrapper
    AND the embedded scene-graph JSON. Highlights the bbox_norm coordinates so the
    perception-conditioning is obvious."""
    text = Path(auto_prompt_path).read_text()
    img = Image.new("RGB", size, (245, 247, 252))
    draw = ImageDraw.Draw(img)
    title = load_font(18)
    code = load_font(13)
    body = load_font(14)

    draw.rectangle([0, 0, size[0], 30], fill=(60, 90, 160))
    draw.text((10, 5), "auto-built prompt  =  natural-language wrapper + scene-graph JSON",
              fill=(255, 255, 255), font=title)

    # Take first non-empty non-JSON line as the wrapper
    wrapper = ""
    for ln in text.splitlines():
        if ln.strip() and not ln.strip().startswith(("{", '"', '}')):
            wrapper = ln.strip()
            break
    y = 38
    # word wrap wrapper at ~80 chars
    if wrapper:
        line = wrapper[:78]
        draw.text((10, y), line, fill=(20, 20, 20), font=body)
        if len(wrapper) > 78:
            draw.text((10, y+18), wrapper[78:156], fill=(20, 20, 20), font=body)
            y += 18
        y += 22

    # Extract JSON portion
    try:
        i_open = text.index("{")
        i_close = text.rindex("}")
        sg_json = json.loads(text[i_open:i_close+1])
    except Exception:
        sg_json = None

    if sg_json is not None:
        # Render the most informative subset of the JSON
        snippet_lines = []
        ws = sg_json.get("workspace") or {}
        if ws.get("bbox_norm"):
            snippet_lines.append('"workspace": {"name": "'+ws.get("name","")[:20]+'", '
                                 '"bbox_norm": '+str([round(v,2) for v in ws["bbox_norm"]])+'}')
        for hand in (sg_json.get("hands") or [])[:2]:
            if hand.get("bbox_norm"):
                snippet_lines.append('"hand":      {"name": "'+hand.get("name","")[:18]+
                                     '", "bbox_norm": '+str([round(v,2) for v in hand["bbox_norm"]])+', '
                                     '"state": "'+hand.get("state","")[:18]+'"}')
        for obj in (sg_json.get("objects") or [])[:4]:
            if obj.get("bbox_norm"):
                snippet_lines.append('"object":    {"name": "'+obj.get("name","")[:18]+
                                     '", "bbox_norm": '+str([round(v,2) for v in obj["bbox_norm"]])+'}')
        relations = (sg_json.get("spatial_constraints") or [])[:3]
        if relations:
            snippet_lines.append('"spatial":  [')
            for r in relations:
                snippet_lines.append('   "'+r[:64]+'",')
            snippet_lines.append(']')

        # JSON area
        draw.rectangle([8, y-2, size[0]-8, y + len(snippet_lines)*16 + 8],
                       fill=(255, 255, 255), outline=(180, 180, 220))
        for ln in snippet_lines:
            if y > size[1] - 30:
                break
            draw.text((14, y+2), ln, fill=(40, 40, 80), font=code)
            y += 16
        y += 12
        draw.text((10, y), "the normalized bboxes (0..1 of frame) anchor the prompt to "
                           "real perception output.",
                  fill=(70, 70, 70), font=body)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exo", required=True)
    ap.add_argument("--dino", required=True)
    ap.add_argument("--scene-graph", required=True)
    ap.add_argument("--auto-prompt", required=True)
    ap.add_argument("--ego", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--depth-cache", default=None)
    ap.add_argument("--title", default="EgoJudge pipeline, end-to-end on one frame")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cell_w, cell_h, label_h, pad = 600, 338, 32, 12
    cols, rows = 4, 2
    top_title_h = 50
    W = cols * cell_w + (cols + 1) * pad
    H = top_title_h + rows * (cell_h + label_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    big = load_font(28)
    lab = load_font(19)

    draw.text((pad+6, 12), args.title, fill=(20, 20, 20), font=big)

    exo   = Image.open(args.exo).convert("RGB")
    dino  = Image.open(args.dino).convert("RGB")
    hands = render_hand_landmarks(args.exo)
    depth = get_depth(args.exo, args.depth_cache)
    sg    = render_scene_graph_overlay(args.exo, args.scene_graph)
    prom  = render_prompt_card(args.auto_prompt, (cell_w, cell_h))
    ego   = Image.open(args.ego).convert("RGB")
    gt    = Image.open(args.gt).convert("RGB") if args.gt and Path(args.gt).exists() else None

    panels = [
        ("1. Source exo",                          exo),
        ("2. Grounding-DINO open-vocab boxes",      dino),
        ("3. MediaPipe hands (21 landmarks/hand)",  hands),
        ("4. Depth Anything V2 depth map",          depth),
        ("5. Auto-built scene graph (bboxes)",      sg),
        ("6. Auto-built prompt = wrapper + JSON",   prom),
        ("7. Generated ego (gpt-image-1 edit)",     ego),
        ("8. Paired GT ego (180 deg corrected)",    gt),
    ]

    for idx, (l, im) in enumerate(panels):
        r, c = divmod(idx, cols)
        x = pad + c * (cell_w + pad)
        y = top_title_h + pad + r * (cell_h + label_h + pad)
        if im is None:
            cb = Image.new("RGB", (cell_w, cell_h), (40, 40, 40))
            ImageDraw.Draw(cb).text((20, 20), "no GT for this frame",
                                    fill=(220, 80, 80), font=lab)
        else:
            im_copy = im.copy(); im_copy.thumbnail((cell_w, cell_h), Image.LANCZOS)
            cb = Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
            cb.paste(im_copy, ((cell_w - im_copy.size[0]) // 2,
                               (cell_h - im_copy.size[1]) // 2))
        canvas.paste(cb, (x, y))
        draw.text((x + 6, y + cell_h + 4), l, fill=(20, 20, 20), font=lab)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(f"wrote {args.out} {canvas.size}")


if __name__ == "__main__":
    main()
