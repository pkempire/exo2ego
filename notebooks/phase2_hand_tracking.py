# ============================================================
# EXO2EGO — PHASE 2: Hand Tracking + Object Segmentation
# MediaPipe Hands (free, no downloads) + Depth Anything V2
# Copy each cell block into Colab, run in order
# Runtime: T4 GPU
# ============================================================

# ---------- CELL 1: Install dependencies ----------
# NOTE: After this cell, restart runtime (Runtime → Restart runtime)
# Then continue from Cell 2. MediaPipe needs restart to load C extension.
!pip install mediapipe==0.10.14 opencv-python pillow matplotlib numpy torch torchvision transformers

# ---------- CELL 2: Imports ----------
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import torch

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

print(f"MediaPipe version: {mp.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")

# ---------- CELL 3: Load Depth Anything V2 (reuse from Phase 1) ----------
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0
)

# ---------- CELL 4: Load a test image (person doing something with hands) ----------
# We'll use a better test image — person reaching for / holding an object
# Feel free to replace with your own image
URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-estimation-example.jpg"
image = Image.open(requests.get(URL, stream=True).raw)
rgb = np.array(image)
print(f"Image shape: {rgb.shape}")

# ---------- CELL 5: Run Depth Anything V2 ----------
result = pipe(image)
depth_map = np.array(result["depth"]).astype(np.float32)
print(f"Depth shape: {depth_map.shape}, range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")

# ---------- CELL 6: Run MediaPipe Hands ----------
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    model_complexity=1  # 1 = full model, 0 = lite
)

results = hands.process(rgb)

if results.multi_hand_landmarks:
    print(f"Detected {len(results.multi_hand_landmarks)} hand(s)")
    for i, hand in enumerate(results.multi_hand_landmarks):
        handedness = results.multi_handedness[i].classification[0].label
        print(f"  Hand {i}: {handedness}, {len(hand.landmark)} landmarks")
else:
    print("No hands detected! Try a different image with visible hands.")

hands.close()

# ---------- CELL 7: Extract 3D hand positions ----------
# MediaPipe gives pixel coords + relative depth. 
# We combine with Depth Anything for metric 3D positions.

def get_hand_3d_positions(rgb, depth_map, hands_result):
    """
    Extract 3D positions of hand keypoints using pixel coords + depth.
    Returns list of {handedness, landmarks_3d, wrist_3d}
    """
    h, w = depth_map.shape
    hands_3d = []
    
    if not hands_result.multi_hand_landmarks:
        return hands_3d
    
    for idx, hand_lms in enumerate(hands_result.multi_hand_landmarks):
        handedness = hands_result.multi_handedness[idx].classification[0].label
        
        landmarks_3d = []
        for lm in hand_lms.landmark:
            # Pixel coordinates
            px = int(lm.x * w)
            py = int(lm.y * h)
            
            # Clamp to image bounds
            px = np.clip(px, 0, w-1)
            py = np.clip(py, 0, h-1)
            
            # Depth from Depth Anything V2
            z = depth_map[py, px]  # metric depth
            
            # MediaPipe relative depth (wrist-relative, normalized)
            z_mp = lm.z
            
            landmarks_3d.append({
                'px': px, 'py': py,
                'depth_metric': float(z),
                'depth_mp': float(z_mp),
            })
        
        # Wrist is landmark 0
        wrist = landmarks_3d[0]
        
        hands_3d.append({
            'handedness': handedness,
            'landmarks_3d': landmarks_3d,
            'wrist_3d': {
                'x': wrist['px'],
                'y': wrist['py'], 
                'z': wrist['depth_metric']
            }
        })
    
    return hands_3d

hands_3d = get_hand_3d_positions(rgb, depth_map, results)

for hand in hands_3d:
    w = hand['wrist_3d']
    print(f"{hand['handedness']} wrist: pixel=({w['x']}, {w['y']}), depth={w['z']:.3f}m")

# ---------- CELL 8: Visualize ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# RGB
axes[0].imshow(rgb)
axes[0].set_title("Input (RGB)", fontsize=14)
axes[0].axis("off")

# Depth
im1 = axes[1].imshow(depth_map, cmap="inferno")
axes[1].set_title("Depth Map", fontsize=14)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Depth + hand landmarks
annotated = rgb.copy()
if results.multi_hand_landmarks:
    for hand_lms in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(
            annotated, hand_lms, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )
        # Highlight wrist (landmark 0)
        wrist = hand_lms.landmark[0]
        wx, wy = int(wrist.x * rgb.shape[1]), int(wrist.y * rgb.shape[0])
        cv2.circle(annotated, (wx, wy), 10, (0, 255, 0), -1)
        cv2.putText(annotated, f"Wrist: {hands_3d[0]['wrist_3d']['z']:.2f}m", 
                    (wx+15, wy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

axes[2].imshow(annotated)
axes[2].set_title("Hand Landmarks + Wrist Depth", fontsize=14)
axes[2].axis("off")

plt.tight_layout()
plt.savefig("phase2_output.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------- CELL 9: Object Segmentation (depth-based) ----------
# Segment the object near the hand using depth thresholding
# This is a simple heuristic — works well for "object in hand" scenarios

def segment_object_near_hand(depth_map, wrist_3d, depth_threshold=0.3):
    """
    Segment the object being held: everything within depth_threshold
    meters of the wrist depth, near the wrist position.
    """
    h, w = depth_map.shape
    wrist_depth = wrist_3d['z']
    wx, wy = wrist_3d['x'], wrist_3d['y']
    
    # Depth-based mask: pixels within threshold of wrist depth
    depth_diff = np.abs(depth_map - wrist_depth)
    depth_mask = depth_diff < depth_threshold
    
    # Spatial mask: within ~150px radius of wrist
    yy, xx = np.mgrid[0:h, 0:w]
    spatial_mask = np.sqrt((xx - wx)**2 + (yy - wy)**2) < 150
    
    # Combined object mask
    object_mask = depth_mask & spatial_mask
    
    # Exclude the hand itself (close to wrist, small region)
    # Simple erosion to remove thin structures (fingers)
    kernel = np.ones((5,5), np.uint8)
    object_mask = cv2.erode(object_mask.astype(np.uint8), kernel, iterations=1)
    object_mask = cv2.dilate(object_mask, kernel, iterations=2)
    
    return object_mask.astype(bool)

if hands_3d:
    obj_mask = segment_object_near_hand(depth_map, hands_3d[0]['wrist_3d'])
    
    # Visualize segmentation
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(rgb)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")
    
    overlay = rgb.copy()
    overlay[obj_mask] = (0, 255, 0)
    blended = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)
    axes[1].imshow(blended)
    axes[1].set_title("Object Segmentation (depth-based)", fontsize=14)
    axes[1].axis("off")
    
    plt.savefig("phase2_segmentation.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # Object depth stats
    obj_depths = depth_map[obj_mask]
    if len(obj_depths) > 0:
        print(f"Object depth: mean={obj_depths.mean():.3f}m, min={obj_depths.min():.3f}m, max={obj_depths.max():.3f}m")
        print(f"Object pixels: {len(obj_depths)}")
else:
    print("No hand detected — skipping object segmentation")

# ---------- CELL 10: Save outputs for Phase 3 ----------
import json

output = {
    'depth_map': depth_map.tolist(),
    'hands': [{
        'handedness': h['handedness'],
        'wrist_pixel': [h['wrist_3d']['x'], h['wrist_3d']['y']],
        'wrist_depth_m': h['wrist_3d']['z'],
        'landmark_count': len(h['landmarks_3d'])
    } for h in hands_3d],
    'image_shape': list(rgb.shape)
}

with open('phase2_output.json', 'w') as f:
    json.dump(output, f, indent=2)

np.save('depth_map.npy', depth_map.astype(np.float32))
np.save('object_mask.npy', obj_mask.astype(bool) if hands_3d else np.zeros_like(depth_map, dtype=bool))

print("Saved phase2_output.json, depth_map.npy, object_mask.npy")

# ---------- CELL 11: What we have now ----------
#
# ✓ Depth map (metric, per-pixel) — from Phase 1
# ✓ Hand landmarks (21 keypoints per hand, 2D pixels) — MediaPipe
# ✓ Wrist 3D position (pixel + metric depth) 
# ✓ Object mask (depth-based segmentation near hand)
#
# Ready for Phase 3: Geometric Reprojection
#   - Backproject depth → 3D point cloud
#   - Place ego camera at wrist/head position
#   - Project point cloud → ego view
#   - Overlay hand mesh (from landmarks)
#   - Overlay object region
