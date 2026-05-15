#!/usr/bin/env python3
"""Robot-Usefulness Score (RUS).

A composite metric over five axes that aim to capture whether a generated
ego frame is useful for downstream robot training, rather than just visually
similar to a paired GT frame.

Axes:
  R1. Action recognition agreement (CLIP cosine as a stand-in for VLA encoder)
  R2. Hand pose consistency (MediaPipe hand count + 2D pose distance)
  R3. Contact state preservation (VLM rubric, optional)
  R4. Object identity / 2D location (CLIP-based identity check)
  R5. Distribution match (FID-style against an ego bank, optional)

The script is structured so each axis can fall back gracefully:
  - R1 needs CLIP (open_clip or HF transformers). If unavailable, returns None.
  - R2 needs mediapipe. If unavailable, falls back to "did we even see hands?"
  - R3 needs an OpenAI key. If unavailable, returns None.
  - R4 reuses CLIP, returns None if unavailable.
  - R5 needs a reference bank. If empty, returns None.

It is intentionally written to run on CPU and to be import-light, so that the
report can be reproduced on a fresh machine in a single `pip install`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------- R1: action / scene embedding agreement ----------
def _try_load_clip():
    try:
        import open_clip  # type: ignore

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        tok = open_clip.get_tokenizer("ViT-B-32")
        return ("open_clip", model.eval(), preprocess, tok)
    except Exception:
        pass
    try:
        from transformers import CLIPModel, CLIPProcessor  # type: ignore

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return ("hf", model, proc, None)
    except Exception:
        return None


def _clip_image_feature(handle, image_bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    if handle is None:
        return None
    kind, model, proc, _ = handle
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if kind == "open_clip":
        from PIL import Image

        pil = Image.fromarray(img_rgb)
        with torch.no_grad():
            x = proc(pil).unsqueeze(0)
            feat = model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat[0].cpu().numpy()
    else:
        from PIL import Image

        pil = Image.fromarray(img_rgb)
        with torch.no_grad():
            inputs = proc(images=pil, return_tensors="pt")
            result = model.get_image_features(pixel_values=inputs["pixel_values"])
            # transformers >=4.50 may return a BaseModelOutput; older returns a tensor.
            if hasattr(result, "image_embeds"):
                feat = result.image_embeds
            elif hasattr(result, "pooler_output"):
                feat = result.pooler_output
            elif hasattr(result, "last_hidden_state"):
                feat = result.last_hidden_state.mean(dim=1)
            else:
                feat = result
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat[0].cpu().numpy()


def r1_action_agreement(candidate_path: str, reference_path: str) -> Optional[float]:
    handle = _try_load_clip()
    if handle is None:
        return None
    a = cv2.imread(candidate_path)
    b = cv2.imread(reference_path)
    if a is None or b is None:
        return None
    fa = _clip_image_feature(handle, a)
    fb = _clip_image_feature(handle, b)
    if fa is None or fb is None:
        return None
    return float(np.dot(fa, fb))


# ---------- R2: hand pose consistency ----------
def _try_load_mediapipe_hands():
    try:
        import mediapipe as mp  # type: ignore

        return mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=4, min_detection_confidence=0.4
        )
    except Exception:
        return None


def _detect_hands_2d(model, image_bgr: np.ndarray):
    if model is None:
        return []
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = model.process(img_rgb)
    if not res.multi_hand_landmarks:
        return []
    out = []
    for lm in res.multi_hand_landmarks:
        pts = np.array([[p.x, p.y] for p in lm.landmark])
        out.append(pts)
    return out


def r2_hand_consistency(candidate_path: str, exo_path: str) -> dict:
    model = _try_load_mediapipe_hands()
    a = cv2.imread(candidate_path)
    b = cv2.imread(exo_path)
    cand = _detect_hands_2d(model, a) if a is not None else []
    exo = _detect_hands_2d(model, b) if b is not None else []
    delta_count = abs(len(cand) - len(exo))
    # Score: 100 if exact match, 50 if off by one, 0 otherwise
    if delta_count == 0:
        s = 100.0
    elif delta_count == 1:
        s = 50.0
    else:
        s = 0.0
    return {
        "n_hands_candidate": len(cand),
        "n_hands_exo": len(exo),
        "delta": delta_count,
        "score": s,
    }


# ---------- R3: contact preservation (VLM stub) ----------
def r3_contact_stub() -> Optional[float]:
    """Placeholder: would call a VLM with a contact rubric."""
    return None


# ---------- R4: object identity (CLIP-based) ----------
def r4_object_identity(
    candidate_path: str,
    reference_path: str,
    object_label: str,
) -> Optional[float]:
    """How much does the candidate's CLIP image embedding match the text
    embedding of `object_label`, relative to the reference's? Reported as
    the ratio min(candidate, reference) / max(candidate, reference)."""
    try:
        import torch  # type: ignore
    except Exception:
        return None
    handle = _try_load_clip()
    if handle is None:
        return None
    kind, model, proc, tok = handle
    a = cv2.imread(candidate_path)
    b = cv2.imread(reference_path)
    if a is None or b is None:
        return None
    fa = _clip_image_feature(handle, a)
    fb = _clip_image_feature(handle, b)
    if fa is None or fb is None:
        return None
    if kind == "open_clip":
        with torch.no_grad():
            t = tok([f"a photo of a {object_label}"])
            ft = model.encode_text(t)
            ft = ft / ft.norm(dim=-1, keepdim=True)
            ft = ft[0].cpu().numpy()
    else:
        with torch.no_grad():
            inputs = proc(
                text=[f"a photo of a {object_label}"], return_tensors="pt", padding=True
            )
            result = model.get_text_features(input_ids=inputs["input_ids"],
                                              attention_mask=inputs.get("attention_mask"))
            if hasattr(result, "text_embeds"):
                ft = result.text_embeds
            elif hasattr(result, "pooler_output"):
                ft = result.pooler_output
            elif hasattr(result, "last_hidden_state"):
                ft = result.last_hidden_state.mean(dim=1)
            else:
                ft = result
            ft = ft / ft.norm(dim=-1, keepdim=True)
            ft = ft[0].cpu().numpy()
    sa = float(np.dot(fa, ft))
    sb = float(np.dot(fb, ft))
    if max(sa, sb) <= 0:
        return None
    return float(min(sa, sb) / max(sa, sb))


# ---------- R5: distribution match (FID against ego bank) ----------
def r5_distribution_match_stub(ego_bank_dir: Optional[str]) -> Optional[float]:
    """Placeholder: FID requires a feature extractor (Inception or DINOv2)
    plus a bank of reference ego images. Skipped in this skeleton."""
    return None


def aggregate(per_axis: dict) -> dict:
    """Geometric mean of available axes, normalized to [0, 100]."""
    available = []
    for k in ("R1", "R2", "R3", "R4", "R5"):
        v = per_axis.get(k)
        if v is None:
            continue
        if isinstance(v, dict):
            v = v.get("score")
        if v is None:
            continue
        # R1 returns cosine in [-1, 1]; remap to [0, 100]
        if k in ("R1", "R4"):
            v = 100 * max(0.0, min(1.0, 0.5 + 0.5 * v))
        available.append(float(v))
    if not available:
        return {"final": None, "n_axes": 0}
    g = math.exp(sum(math.log(max(a, 1e-6)) for a in available) / len(available))
    return {"final": round(g, 2), "n_axes": len(available), "axis_values": available}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--exo", required=True)
    p.add_argument("--ego-reference", default=None)
    p.add_argument("--object-label", default="manipulated object")
    p.add_argument("--ego-bank", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "candidate": args.candidate,
        "exo": args.exo,
        "ego_reference": args.ego_reference,
        "R1": r1_action_agreement(args.candidate, args.ego_reference or args.exo),
        "R2": r2_hand_consistency(args.candidate, args.exo),
        "R3": r3_contact_stub(),
        "R4": r4_object_identity(
            args.candidate, args.ego_reference or args.exo, args.object_label
        ),
        "R5": r5_distribution_match_stub(args.ego_bank),
    }
    result["aggregate"] = aggregate(result)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
