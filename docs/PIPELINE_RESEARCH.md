# Pipeline Component Selection — Exo2Ego

> Last updated: 2026-05-07

## Selected Models (v1)

| Stage | Model | Release | Why |
|-------|-------|---------|-----|
| Depth | **Depth Anything V2 (Large)** | NeurIPS 2024 | 335M params, metric depth fine-tuned, 10x faster than SD-based, HuggingFace available |
| Segmentation | **SAM 3.1** | Mar 2026 | Text + visual prompts, object multiplex (7x faster), handles occlusion, SOTA video tracking |
| Hand Pose | **HaMeR** | CVPR 2024 | ViT-based, 2nd place Ego-Exo4D hands challenge, MANO mesh output, handles occlusion |
| Object Pose | **FoundationPose** | 2024 | Zero-shot 6-DoF on novel objects, no CAD needed, robust to occlusion |
| Camera | **Depth Anything V2 + COLMAP** | — | Depth for scale, COLMAP for extrinsics on moving cameras |
| Diffusion | **Exo2Ego-V** (optional) | NeurIPS 2024 | Specifically designed for exo→ego, video-consistent |
| Captioning | **VLM (GPT-4o or Florence-2)** | — | Auto-generate verb/noun/caption from frames |

## Pipeline Details

### 1. Depth Estimation — Depth Anything V2

**Input:** RGB frame (any resolution)
**Output:** Metric depth map (same resolution), or relative depth + scale factor

**Setup:**
```python
from transformers import pipeline
depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
result = depth_pipe(image)
depth_map = result["predicted_depth"]  # (H, W) float32 tensor
```

**Variants (choose based on hardware):**
| Variant | Params | Speed | Use Case |
|---------|--------|-------|----------|
| Small | 24.8M | Fastest | Real-time, edge |
| Base | 97.5M | Fast | Good desktop |
| Large | 335.3M | Moderate | Best quality |
| Giant | 1.3B | Slow | Research |

**Metric depth:** Fine-tuned versions available for NYUv2 (indoor). Use this for tabletop scenes.

### 2. Segmentation — SAM 3.1

**Input:** Image + text prompt (e.g., "hand", "screwdriver", "table")
**Output:** Segmentation masks with instance IDs, tracks across video frames

**Setup:**
```python
# SAM 3.1 with object multiplex
from sam3 import SAM3VideoPredictor
predictor = SAM3VideoPredictor.from_pretrained("facebook/sam3.1")
# Text-prompted segmentation across video
masks = predictor.segment_video(
    video_frames,
    prompts=["hand", "tool", "object"],
    mode="text"
)
```

**Key features for our use case:**
- Open-vocabulary: prompt with any object name
- Video tracking: same object gets consistent ID across frames
- Object multiplex: process 16 objects in single forward pass
- Handles occlusion and reappearance

### 3. Hand Pose — HaMeR

**Input:** Cropped hand image + hand side (left/right)
**Output:** MANO hand mesh (778 vertices, 21 joints), 3D keypoints

**Setup:**
```python
# HaMeR inference
from hamer import HaMeR
model = HaMeR.from_pretrained("geopavlakos/hamer")
# Process per-frame
hand_output = model.predict(
    image=cropped_hand,
    hand_side="right"
)
# Output: MANO vertices, 3D joints, camera params
vertices = hand_output["vertices"]  # (778, 3)
joints_3d = hand_output["joints"]   # (21, 3)
```

**Why HaMeR over MediaPipe:**
- SOTA accuracy on Ego-Exo4D hands benchmark
- Robust to occlusion, gloves, diverse skin tones
- Better temporal consistency (even per-frame)
- Directly outputs MANO mesh (standard format)

**Dependency:** Needs hand detection first (SAM 3.1's "hand" text prompt handles this)

### 4. Object Pose — FoundationPose

**Input:** RGB-D image + object mask + 3D model (optional — can use a reference image instead)
**Output:** 6-DoF object pose (R, t)

**Alternative if FoundationPose is heavy:** Use SAM 3.1 to segment objects, then use a simpler keypoint detector for known object categories.

**Key consideration:** For tabletop assembly with known tools (screwdriver, wrench, etc.), we can use category-level pose estimation which is simpler than instance-level.

### 5. Reprojection — Custom

**Algorithm:**
```
1. From depth + segmentation, build a point cloud of the scene
2. Estimate actor's head position from hand pose + body heuristics
3. Place virtual camera at head position, facing the hands
4. Project point cloud into virtual camera view
5. For occluded regions: mark as "unknown" (to be filled by diffusion)
6. Output: geometrically correct ego frame + occlusion mask
```

**Implementation:** NumPy + OpenCV. ~200 lines of Python. No ML model needed.

### 6. Diffusion Cleanup — Exo2Ego-V (Optional for v1)

**When to use:** If reprojected frames look too distorted for downstream use.

**When to skip:** If geometric correctness is sufficient (likely for VLA pre-training where models need object position more than photorealistic textures).

### 7. Annotation — VLM Auto-Captioning

**Input:** Generated ego frame  
**Output:** `{verb, noun, caption, contact_flag}`

Use GPT-4o or Florence-2 to auto-label frames. This adds weak supervision that VLA models can use.

## Evaluation Plan

### Qualitative
- Side-by-side: exo input → reprojected ego → diffusion-cleaned ego
- Manual inspection of hand/object placement accuracy

### Quantitative
- Reprojection error on Ego-Exo4D paired data (ground truth ego exists)
- Depth consistency between exo and generated ego views
- Hand keypoint accuracy vs ground truth ego

### Downstream (stretch)
- Train simple behavior cloning policy on generated data
- Compare success rate vs. policy trained on real ego data only

## Dependencies

```
torch >= 2.0
transformers >= 4.40
opencv-python >= 4.8
sam3  # Meta SAM 3.1
hamer  # HaMeR hand mesh recovery
numpy
pillow
tqdm
```

## Hardware Requirements

- GPU with 8GB+ VRAM (for SAM 3.1 + Depth Anything V2)
- Models can run sequentially (don't need all loaded simultaneously)
- Apple Silicon works for Depth Anything V2 via Core ML
