#!/usr/bin/env python3
"""Phase 2: Hand tracking + object segmentation (local)."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")

# --- DEPENDENCIES ---
# pip install mediapipe opencv-python pillow matplotlib numpy torch torchvision transformers

import cv2, numpy as np, mediapipe as mp
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt
import torch

# --- CONFIG ---
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else None

# --- LOAD MODELS ---
# Depth Anything V2-Base: produces METRIC depth, not relative
# Small model: relative depth (arbitrary units) — USELESS for reprojection
# Base model: metric depth — WHAT WE NEED
MODEL = "depth-anything/Depth-Anything-V2-Base-hf"
print(f"Loading {MODEL} (metric depth)...")
device = 0 if torch.cuda.is_available() else (-1 if not torch.backends.mps.is_available() else "mps")
pipe = pipeline("depth-estimation", model=MODEL, device=device)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# --- LOAD IMAGE ---
if IMAGE_PATH:
    rgb = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
else:
    print("No image path given. Drag an image file onto this script or pass as argument.")
    print("Usage: python3 phase2_local.py /path/to/image.jpg")
    sys.exit(1)

print(f"Image: {rgb.shape}")

# --- DEPTH ---
result = pipe(image)
# Depth Anything V2-Small: use predicted_depth tensor, not the PIL image
predicted_depth = result["predicted_depth"].detach().cpu().numpy() if hasattr(result["predicted_depth"], 'detach') else np.array(result["predicted_depth"])
depth_map = predicted_depth.astype(np.float32) if predicted_depth.ndim == 2 else predicted_depth.squeeze()
print(f"Depth: {depth_map.shape}, range [{depth_map.min():.2f}, {depth_map.max():.2f}]")

# --- HANDS ---
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, model_complexity=1)
results = hands.process(rgb)
hands.close()

if not results.multi_hand_landmarks:
    print("NO HANDS DETECTED — use an image with visible hands.")
    sys.exit(0)

print(f"Found {len(results.multi_hand_landmarks)} hand(s)")

# --- EXTRACT 3D POSITIONS ---
h, w = depth_map.shape
hands_3d = []

for idx, hand_lms in enumerate(results.multi_hand_landmarks):
    handedness = results.multi_handedness[idx].classification[0].label
    landmarks = []
    for lm in hand_lms.landmark:
        px, py = np.clip(int(lm.x * w), 0, w-1), np.clip(int(lm.y * h), 0, h-1)
        landmarks.append({"px": px, "py": py, "depth_m": float(depth_map[py, px]), "z_mp": float(lm.z)})
    
    wrist = landmarks[0]
    hands_3d.append({"handedness": handedness, "landmarks": landmarks, "wrist": wrist})
    print(f"  {handedness} wrist: px=({wrist['px']},{wrist['py']}) depth={wrist['depth_m']:.3f}m")

# --- OBJECT SEGMENTATION ---
for hand in hands_3d:
    wrist = hand["wrist"]
    depth_diff = np.abs(depth_map - wrist["depth_m"])
    depth_mask = depth_diff < 0.3
    yy, xx = np.mgrid[0:h, 0:w]
    spatial = np.sqrt((xx - wrist["px"])**2 + (yy - wrist["py"])**2) < 150
    obj_mask = (depth_mask & spatial).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    obj_mask = cv2.dilate(cv2.erode(obj_mask, kernel, iterations=1), kernel, iterations=2)
    hand["object_mask"] = obj_mask.astype(bool)
    obj_depths = depth_map[obj_mask.astype(bool)]
    if len(obj_depths) > 0:
        print(f"  Object: {len(obj_depths)} px, depth {obj_depths.mean():.3f}±{obj_depths.std():.3f}m")

# --- VISUALIZE ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")
im = axes[1].imshow(depth_map, cmap="inferno"); axes[1].set_title("Depth"); axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046)

annotated = rgb.copy()
if results.multi_hand_landmarks:
    for hand_lms in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(annotated, hand_lms, mp_hands.HAND_CONNECTIONS,
                               mp_styles.get_default_hand_landmarks_style(),
                               mp_styles.get_default_hand_connections_style())
for hand in hands_3d:
    wx, wy = hand["wrist"]["px"], hand["wrist"]["py"]
    cv2.circle(annotated, (wx, wy), 10, (0,255,0), -1)
    cv2.putText(annotated, f"{hand['wrist']['depth_m']:.2f}m", (wx+15, wy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    annotated[hand["object_mask"]] = annotated[hand["object_mask"]] * 0.6 + np.array([0,255,0]) * 0.4

axes[2].imshow(annotated); axes[2].set_title("Hands + Object"); axes[2].axis("off")
plt.tight_layout()
out_img = IMAGE_PATH.rsplit(".",1)[0] + "_phase2.png"
plt.savefig(out_img, dpi=150, bbox_inches="tight")
print(f"Saved: {out_img}")

# --- SAVE DATA ---
out = {"image_shape": list(rgb.shape), "rgb": rgb.tolist(), "depth_map": depth_map.tolist(), "hands": []}
for hand in hands_3d:
    out["hands"].append({
        "handedness": hand["handedness"],
        "wrist_pixel": [int(hand["wrist"]["px"]), int(hand["wrist"]["py"])],
        "wrist_depth_m": float(hand["wrist"]["depth_m"]),
        "object_pixel_count": int(hand["object_mask"].sum()),
        "object_mean_depth": float(depth_map[hand["object_mask"]].mean()) if hand["object_mask"].sum() > 0 else 0.0
    })

json_path = IMAGE_PATH.rsplit(".",1)[0] + "_phase2.json"
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved: {json_path}")
print("Done. Ready for Phase 3.")
