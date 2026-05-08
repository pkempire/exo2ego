"""Phase 2: Hand Tracking + Object Segmentation — runs locally on Mac."""
import cv2, numpy as np, json, mediapipe as mp
from PIL import Image
from transformers import pipeline
import torch

# ─── Load Depth Anything V2 ───
print("Loading Depth Anything V2...")
pipe = pipeline("depth-estimation", "depth-anything/Depth-Anything-V2-Small-hf", 
                device=0 if torch.cuda.is_available() else -1)
print("Done.")

# ─── Load test image ───
image = Image.open("test_frame.png")  # PUT YOUR FRAME HERE
rgb = np.array(image)
print(f"Image: {rgb.shape}")

# ─── Depth ───
depth = np.array(pipe(image)["depth"]).astype(np.float32)

# ─── Hands ───
hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, 
                                  min_detection_confidence=0.5, model_complexity=1)
result = hands.process(rgb)
hands.close()

if result.multi_hand_landmarks:
    for i, h in enumerate(result.multi_hand_landmarks):
        w = h.landmark[0]  # wrist
        px, py = int(w.x * rgb.shape[1]), int(w.y * rgb.shape[0])
        z = float(depth[min(py, depth.shape[0]-1), min(px, depth.shape[1]-1)])
        handedness = result.multi_handedness[i].classification[0].label
        print(f"{handedness} wrist: ({px},{py}) depth={z:.3f}m")
else:
    print("No hands detected — use an image with visible hands")

# ─── Save ───
np.save("depth_map.npy", depth)
print("Saved depth_map.npy")
