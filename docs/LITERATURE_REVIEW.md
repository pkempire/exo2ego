# Exo2Ego — Literature Review & Build Plan

> Written from actually reading papers and repos, not summaries.  
> Parth Kocheta — May 2026

---

## 1. What Everyone Else Does (And Why We're Doing Something Different)

### 1.1 Exo2Ego-V (NeurIPS 2024) — The Expensive Way

**What it does:** Takes FOUR calibrated exocentric cameras placed 360° around a scene, plus a known egocentric camera pose, and generates an egocentric video using a trained diffusion model.

**How it works:**
1. Multi-view exo encoder: process 4 camera views through separate pose-conditioned encoders, fuse into dense multi-scale features
2. View translation prior: a small network that maps exo features → spatially aligned ego features using known camera extrinsics
3. Ego video diffusion: a video diffusion model (based on MagicAnimate) conditioned on the translated features + temporal attention

**Training:** Two-stage. Stage 1 trains spatial appearance generation. Stage 2 adds temporal layers for video consistency. Trained on Ego-Exo4D.

**The problem for us:**
- Needs 4 calibrated cameras (we have ONE uncalibrated phone video)
- Needs known ego camera pose (we have to ESTIMATE it)
- Needs training a diffusion model on Ego-Exo4D (compute-heavy)
- Produces video output (we just need frames)

**Verdict:** NOT reusable for us. The approach is too constrained and compute-heavy. But the evaluation metrics (LPIPS, FID) and the Ego-Exo4D dataset are useful.

**Code:** github.com/showlab/Exo2Ego-V (58 stars, Apache 2.0, Python)

### 1.2 EgoWorld (2025) — The Right Philosophy

**What it does:** Single exo view → point cloud from depth → reproject to ego viewpoint → diffusion cleanup → ego frame. Exactly our philosophy.

**How it works:**
1. Estimate depth map from exo RGB using a pretrained depth model
2. Back-project depth + RGB into a 3D point cloud
3. Estimate hand pose (3D keypoints) from the exo view
4. Place virtual camera at the estimated head/eye position
5. Reproject point cloud into virtual camera = sparse ego view
6. Feed sparse reprojection + hand pose guidance into a diffusion inpainting model
7. Output: dense, photorealistic ego frame

**Key insight I missed earlier:** EgoWorld shows that the reprojected frame doesn't need to be perfect — it just needs to get the geometry right. The diffusion model handles the "filling in" of missing pixels and photorealism. THIS is the right approach.

**The problem for us:**
- No open-source code (paper only)
- Still uses a diffusion model (but a lightweight one, not full video generation)
- The hand-pose-guided inpainting is the novel part

**Verdict:** This is the template for our approach. But we can simplify further by skipping the diffusion step and just delivering geometrically-correct reprojected frames with annotations.

**Code:** Not open source. Paper at arxiv.org/abs/2506.17896

### 1.3 VideoMimic (CoRL 2025) — Real-to-Sim, Not Our Problem

**What it does:** Takes a video of a human doing something → reconstructs 3D human pose + scene geometry → retargets motion to humanoid robot → trains control policy in simulation.

**How it works:**
1. Structure from Motion (COLMAP) to recover camera + sparse scene
2. Human pose estimation (WHAM, SMPL) to get 3D body
3. Scene reconstruction: dense point cloud → mesh
4. Motion retargeting: human SMPL → robot configuration
5. Sim training: reference motion tracking → RL fine-tuning
6. Real robot deployment

**What's relevant to us:** Stage 1-2 (SfM + pose estimation). Their reconstruction pipeline shows it's possible to get metric-scale 3D from a single video. Their ego-view rendering at the end is a bonus they mention but don't focus on.

**What's NOT relevant:** Stages 3-6. We don't need robot policy training — we're producing the DATA that feeds into policy training.

**Verdict:** Use their SfM + pose approach as reference. Their real-to-sim pipeline is overkill for us.

**Code:** github.com/hongsukchoi/VideoMimic (768 stars, MIT, Python + C++)

### 1.4 PhysBrain (Physical Intelligence, Dec 2025) — The Business Case

**What it does:** Takes raw human egocentric videos → multi-level structured supervision for VLAs. Proves that human ego data improves robot policies ~2x.

**How it works:**
1. Egocentric2Embodiment Translation Pipeline: raw ego video → schema-driven annotations
2. Structured supervision: multi-level labeling (actions, objects, contacts)
3. Trains PhysBrain: an egocentric-aware VLM for planning
4. Fine-tune VLA on PhysBrain initialization for specific tasks

**Key takeaway for us:** The data format matters. Their schema (action labels, object states, temporal structure) is what makes the data useful for VLAs. Our output should follow a similar structured format, not just raw frames.

**Verdict:** Validates the entire premise. PI explicitly says human egocentric data is valuable and scarce. We're building a pipeline to CREATE this data from abundant exo video.

**Code:** Not open source. Paper at arxiv.org/abs/2512.16793

### 1.5 SceneComplete (2024) — Modular Perception Composition

**What it does:** Takes a single RGB-D image of a cluttered scene and produces a complete, segmented 3D model by composing pretrained modules.

**How it works:**
1. VLM (GPT-4V): describe the scene, identify objects
2. SAM: segment individual objects from the description
3. Image inpainting (LaMa): fill in occluded regions behind each object
4. Image-to-3D (Zero-1-to-3): generate 3D model for each completed object
5. Pose estimation: place objects back in the scene
6. Output: complete 3D scene model with all objects

**Why this matters:** This is PROOF that composing pretrained models works for perception tasks. They don't train anything — they just chain existing models. Their VLM → segmentation → inpainting → 3D pipeline is conceptually identical to our depth → segmentation → pose → reprojection pipeline.

**Verdict:** Methodological validation. We're doing the same thing for a different output (ego frames instead of 3D models).

**Code:** Not open source (as of what I found). Paper at arxiv.org/abs/2410.23643

---

## 2. The Models We'll Actually Use

### 2.1 Depth Anything V2 (NeurIPS 2024)

**What it actually is:** A DINOv2-Giant ViT backbone with a DPT decoder head, trained on 595K synthetic labeled images + 62M pseudo-labeled real images. The "V2" improvement over V1: replaced all real labeled data with synthetic (more precise), scaled teacher model capacity, and used pseudo-labels as the bridge from teacher to student.

**Why it's good:** Monocular depth (single image → depth map). No stereo, no multi-view, no camera parameters needed. Works on any image. Metric depth variant available (trained on NYUv2 for indoor scenes — exactly our use case).

**How to use it (Colab):**
```python
# Option A: HuggingFace pipeline (easiest)
from transformers import pipeline
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
result = pipe(image)  # returns {"predicted_depth": tensor(H,W), "depth": PIL image}
```

```python
# Option B: Direct model for metric depth
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
```

**Compute:** Works on T4 GPU (Colab free). Large model ~2GB VRAM. Inference: ~100ms per frame.

### 2.2 SAM 3.1 (Meta, March 2026)

**What it actually is:** A unified model for detection + segmentation + tracking. Built on a ViT backbone shared across all tasks. The 3.1 update added Object Multiplex: process up to 16 objects in a single forward pass by sharing memory.

**How it works for segmentation:**
- Image encoder: ViT processes the image once, produces feature embeddings
- Prompt encoder: encodes text prompts (e.g., "hand", "screwdriver") or click/box prompts
- Mask decoder: lightweight transformer that produces segmentation masks from image features + prompt embeddings
- For video: adds a memory bank that stores previous frame features for temporal consistency

**Why it's good for us:** We can prompt with text ("hand", "tool", "table") and get instance masks. Handles occlusion. Tracks objects across frames. Open-vocabulary means we don't need to train on specific objects.

**How to use it (Colab):**
```python
from sam3 import build_sam3_video_predictor
predictor = build_sam3_video_predictor()
# For image segmentation with text prompts
masks = predictor.segment_image(
    image,
    prompts=["hand", "screwdriver", "table surface"],
    mode="text"
)
# For video: add frames sequentially
for frame in frames:
    predictor.add_frame(frame)
masks = predictor.get_tracked_masks()
```

**Compute:** Needs GPU. SAM 3.1 works on T4 but is slower than H100. ~2-3 FPS on T4, ~32 FPS on H100.

**Important constraint:** SAM 3.1 model weights are gated on HuggingFace. Need to request access from Meta.

### 2.3 HaMeR (CVPR 2024)

**What it actually is:** A ViT-Huge backbone that takes a cropped hand image → predicts MANO hand model parameters (pose + shape) → outputs 3D mesh (778 vertices) and 21 3D joints. Trained on a combination of 2D and 3D hand datasets.

**How MANO works:** MANO (hand Model with Articulated and Non-rigid deformations) is a parametric model. It takes pose parameters (joint angles, 45 DoF) + shape parameters (10 DoF) → outputs a 3D mesh. HaMeR learns to predict these parameters from an image.

**Why it's good for us:** SOTA accuracy. Handles occlusion (second place in Ego-Exo4D hands challenge). Outputs MANO format which is standard and easy to convert to keypoints. Works per-frame with good temporal consistency.

**How to use it (Colab):**
```python
# HaMeR needs: cropped hand image + hand side (left/right)
from hamer import HaMeR
model = HaMeR.from_pretrained("geopavlakos/hamer")

# Step 1: Detect hand bounding box (use SAM 3.1 for this)
# Step 2: Run HaMeR on cropped hand
output = model.predict(
    image=cropped_hand_image,
    hand_side="right"
)
# output["vertices"]: (778, 3) — 3D mesh vertices
# output["joints"]: (21, 3) — 3D joint positions  
# output["keypoints_2d"]: (21, 2) — projected 2D keypoints
```

**Compute:** Works on T4. ~50ms per hand per frame.

### 2.4 ChronoDepth / Video Depth (Optional)

We don't NEED temporally consistent depth (each frame is reprojected independently). But if we want smooth depth across frames, ChronoDepth (arxiv 2406.01493) uses video diffusion priors for cross-frame consistency. Skip for v1.

---

## 3. Our Actual Pipeline (Revised After Research)

### Frame-by-frame, not video-to-video

Trying to generate temporally consistent ego video is the WRONG approach for a class project. VLA models process individual frames — they don't need video. Frame-by-frame is:
- 10x simpler
- No temporal consistency problems
- Each frame is independently evaluable
- We can parallelize across frames

### Revised Pipeline:

```
INPUT: Single third-person RGB frame from a phone video of tabletop task
OUTPUT: Egocentric RGB frame with annotations

STAGE 1: DEPTH
  Model: Depth Anything V2 Metric Indoor Large
  Input: RGB frame
  Output: Metric depth map (meters)
  Why: We need real-world scale for accurate reprojection

STAGE 2: SEGMENTATION  
  Model: SAM 3.1
  Input: RGB frame + text prompts ["hand", "tool", "object", "table"]
  Output: Instance masks for hands, objects, table surface
  Why: We need to know what's what for pose estimation and occlusion

STAGE 3: HAND POSE
  Model: HaMeR
  Input: Cropped hand regions (from SAM masks) + hand side
  Output: 3D hand mesh (MANO), 21 3D joints
  Why: Critical for placing the virtual ego camera and understanding manipulation

STAGE 4: REPROJECTION (The Core)
  Algorithm: Custom (OpenCV + NumPy)
  1. From depth map + RGB, build 3D point cloud: P_3d = depth * K^(-1) * pixel
  2. Place virtual camera at actor's head position:
     - head_pos ≈ hand_pos + [0, 0.3, 0] (30cm above hands, facing forward)
     - look_at = midpoint of hands
  3. Compute virtual camera extrinsics (R_v, t_v)
  4. Project point cloud into virtual camera:
     pixel_ego = K_v * [R_v | t_v] * P_3d
  5. Handle occlusion via z-buffering
  6. Output: sparse ego frame + occlusion mask (which pixels are unknown)

STAGE 5: OPTIONAL CLEANUP
  Model: LaMa (Large Mask Inpainting) or Stable Diffusion Inpainting
  Input: Sparse ego frame + occlusion mask
  Output: Filled-in ego frame
  Why: Make it visually presentable. NOT required for VLA training.
  Skip for v1.

STAGE 6: ANNOTATION
  Model: Florence-2 or GPT-4o
  Input: Generated ego frame
  Output: {"verb": "...", "noun": "...", "caption": "...", "contact": bool}
```

### Key Design Decisions:

| Decision | Why |
|----------|-----|
| Frame-by-frame | VLA models process frames, not video. Temporal consistency is unnecessary. |
| Metric depth | We need real scale for accurate reprojection. NYUv2 indoor fine-tuned. |
| No diffusion required | Geometric correctness > photorealism for VLA training |
| All pretrained models | Zero training. Just inference. |
| Colab-compatible | Everything runs on free T4 GPU |
| Structured output | Following PhysBrain's schema pattern for VLA compatibility |

---

## 4. Key Papers (With My Understanding)

### Must-Read (understand these deeply):
1. **Ego-Exo4D** (Grauman et al., CVPR 2024) — The dataset. 1,286 hours of paired ego+exo video across 839 participants doing skilled tasks. The benchmark for this problem.
2. **Exo2Ego-V** (Liu et al., NeurIPS 2024) — The SOTA diffusion approach. Read to understand what NOT to do.
3. **EgoWorld** (Park et al., 2025) — The modular approach. Closest to ours. Read for the reprojection + inpainting idea.
4. **π0.5** (Physical Intelligence, 2025) — Why this data matters. VLAs need heterogeneous data.
5. **PhysBrain** (Physical Intelligence, 2025) — The data format. Schema for VLA-compatible annotations.

### Read for Method:
6. **Depth Anything V2** (Yang et al., NeurIPS 2024) — Our depth model. Synthetic data + teacher scaling.
7. **SAM 3** (Carion et al., 2025) — Our segmentation model. Concept prompts.
8. **HaMeR** (Pavlakos et al., CVPR 2024) — Our hand pose model. ViT scaling.

### Read for Context:
9. **VideoMimic** (Allshire & Choi et al., CoRL 2025) — Real-to-sim from internet video. Related but different problem.
10. **SceneComplete** (Agarwal et al., 2024) — Modular perception composition. Methodological inspiration.

---

## 5. Build Plan (Concrete Steps)

### Week 1: Single Frame Pipeline
- [ ] Download one Ego-Exo4D clip (paired exo + ego ground truth)
- [ ] Stage 1: Run Depth Anything V2 on a single exo frame → depth map
- [ ] Stage 2: Run SAM 3.1 with text prompts → hand + object masks
- [ ] Verify: hand detection from SAM is good enough for HaMeR
- [ ] Stage 3: Run HaMeR on cropped hands → 3D hand mesh
- [ ] Stage 4: Implement reprojection (custom code)
- [ ] Compare: reprojected ego frame vs ground truth ego frame

### Week 2: Pipeline Refinement + Annotation
- [ ] Tune reprojection parameters (head position estimation, camera intrinsics)
- [ ] Stage 5 (optional): Try LaMa inpainting for visual quality
- [ ] Stage 6: Add VLM captioning
- [ ] Generate 10 annotated ego frames from different exo views
- [ ] Write qualitative evaluation

### Week 3: Paper
- [ ] Write method section with architecture diagrams
- [ ] Run evaluation metrics (LPIPS vs ground truth ego)
- [ ] Ablation: with vs without each stage
- [ ] Write results, discussion, conclusion

---

## 6. Compute Strategy

**No local GPU needed.** Everything runs in Google Colab:
- Free tier: T4 GPU, 12GB RAM, ~4hr session limit
- Models fit: Depth Anything V2 (~2GB), SAM 3.1 (~4GB), HaMeR (~2GB)
- Run sequentially (not simultaneously) to stay within VRAM
- Use HuggingFace model caching to avoid re-downloading

**Alternative:** HuggingFace Inference API (paid, per-call) or Replicate (per-run). Not needed for our scale.

---

## 7. Existing Code We Can Use

| Component | Source | License | Notes |
|-----------|--------|---------|-------|
| Depth Anything V2 | HuggingFace | Apache 2.0 | One-line pipeline |
| SAM 3.1 | github.com/facebookresearch/sam3 | Custom (Meta) | Gated weights |
| HaMeR | github.com/geopavlakos/hamer | MIT? | Pretrained weights |
| Reprojection | Custom | MIT | We write this (~200 lines) |
| LaMa inpainting | github.com/advimman/lama | Apache 2.0 | Optional |

**Nothing out there does exactly what we're building.** Everyone either does full diffusion (Exo2Ego-V), full real-to-sim (VideoMimic), or doesn't open-source (EgoWorld, SceneComplete). Our modular perception pipeline is novel in its simplicity.

---

*This is a living document. Update as we build and learn.*
