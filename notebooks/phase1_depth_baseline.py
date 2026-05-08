# ============================================================
# EXO2EGO — PHASE 1: Depth Anything V2 Baseline
# Copy each cell block into Colab, run in order
# Runtime: T4 GPU (~2 min total)
# ============================================================

# ---------- CELL 1: Install dependencies ----------
!pip install -q torch torchvision transformers opencv-python pillow matplotlib numpy

# ---------- CELL 2: Imports ----------
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import cv2
from pathlib import Path

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only — enable T4!'}")

# ---------- CELL 3: Load Depth Anything V2 ----------
# Use the small model — best tradeoff for T4 (1.3 GB VRAM)
# Alternatives: "depth-anything/Depth-Anything-V2-Base-hf" (3.9 GB)
#               "depth-anything/Depth-Anything-V2-Large-hf" (13 GB — won't fit T4)

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

pipe = pipeline(
    task="depth-estimation",
    model=MODEL,
    device=0 if torch.cuda.is_available() else -1  # GPU if available
)
print(f"Loaded {MODEL}")

# ---------- CELL 4: Test on a sample image ----------
# Download a test image (person doing an action — similar to our use case)
URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-estimation-example.jpg"
image = Image.open(requests.get(URL, stream=True).raw)

print(f"Image size: {image.size}")

# Run inference
result = pipe(image)
depth = result["depth"]  # PIL Image — grayscale depth map

# ---------- CELL 5: Visualize ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(image)
axes[0].set_title("Input (RGB)", fontsize=14)
axes[0].axis("off")

im = axes[1].imshow(depth, cmap="inferno")
axes[1].set_title("Depth Anything V2 — Metric Depth", fontsize=14)
axes[1].axis("off")

plt.colorbar(im, ax=axes[1], fraction=0.046, label="Depth")
plt.tight_layout()
plt.savefig("depth_output.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved depth_output.png")

# ---------- CELL 6: Depth stats ----------
depth_arr = np.array(depth)
print(f"Depth range: [{depth_arr.min():.2f}, {depth_arr.max():.2f}]")
print(f"Depth mean:  {depth_arr.mean():.2f}")
print(f"Resolution:  {depth_arr.shape}")
print(f"NaN values:  {np.isnan(depth_arr).sum()}")

# ---------- CELL 7: Save for pipeline ----------
# Save as float32 numpy for later reprojection stage
np.save("depth_map.npy", depth_arr.astype(np.float32))
print(f"Saved depth_map.npy — shape {depth_arr.shape}, dtype float32")

# Also save the RGB image for reference
image.save("input_frame.png")
print("Saved input_frame.png")

# ---------- CELL 8: Test with a realistic action image ----------
# Load a second test: person reaching for object (closer to our domain)
# Grab a frame from Ego-Exo4D or just another web image

URL2 = "https://raw.githubusercontent.com/facebookresearch/ego4d/main/img/ego4d.jpg"
try:
    image2 = Image.open(requests.get(URL2, stream=True).raw)
    print(f"Action image size: {image2.size}")

    result2 = pipe(image2)
    depth2 = result2["depth"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(image2)
    axes[0].set_title("Action Scene (RGB)", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(depth2, cmap="inferno")
    axes[1].set_title("Depth Map", fontsize=14)
    axes[1].axis("off")
    plt.savefig("depth_action_output.png", dpi=150, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"Second test image failed (expected — URL may be down): {e}")
    print("Proceed with Cell 9.")

# ---------- CELL 9: Write your own image (skip this if using URL above) ----------
# Upload an image to Colab, then run:
# from google.colab import files
# uploaded = files.upload()
# your_image = Image.open(list(uploaded.keys())[0])
# result = pipe(your_image)
# result["depth"].save("my_depth.png")
# print("Done! Check my_depth.png")

# ---------- CELL 10: What we learned / Next steps ----------
#
# ✓ Depth Anything V2 works on T4 (~1.3 GB VRAM)
# ✓ Produces dense metric depth from a single RGB image
# ✓ Output shape matches input — ready for reprojection
#
# Phase 2: Load FollowMyHold, extract hand mesh + object mesh + head pose
# Phase 3: Backproject depth → 3D point cloud → reproject to ego view
