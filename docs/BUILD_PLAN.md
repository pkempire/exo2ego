# Exo2Ego — Updated Build Plan (May 7, 2026)

## What Changed After Deep Research

Two critical discoveries:

1. **FollowMyHold** (3DV 2026, MIT): Already does exactly what we need for hand-object pose estimation. Uses HaMeR + MoGe + Hunyuan3D + geometric optimization. 57 stars, full code released Jan 2026.

2. **HOLD** (CVPR 2024 Highlight, 471 stars, MIT): Joint hand-object reconstruction from monocular video. Actively maintained (last push March 2026). Template-free.

This means we DON'T need to build hand pose + object pose from scratch. We can use FollowMyHold as the foundation and focus on our novel contribution: exo→ego reprojection.

## Revised Pipeline

```
Stage 0: FollowMyHold → hand mesh + object mesh + scene geometry
Stage 1: Depth Anything V2 → metric depth map
Stage 2: Reprojection → virtual ego camera view (OUR NOVEL CONTRIBUTION)
Stage 3: ProPainter → fill occluded regions using temporal context
Stage 4: Wan2.2 TI2V-5B → realistic ego video clip (Colab Pro)
Stage 5: Annotation → VLM auto-captioning
```

## Stage-by-Stage Build Plan

### Stage 0: Hand-Object Reconstruction (Use FollowMyHold)
**Input:** Monocular video of tabletop task
**Output:** 3D hand mesh (MANO), object meshes, camera trajectory
**Model:** github.com/aidilayce/FollowMyHold (MIT, Python)
**Dependencies:** HaMeR, MoGe, Hunyuan3D-2
**Compute:** Colab Pro (needs GPU, ~16GB VRAM)
**Risk:** Medium — needs to work on our specific video types

### Stage 1: Depth Estimation 
**Input:** RGB frame
**Output:** Metric depth map
**Model:** Depth Anything V2 Metric Indoor Large (HuggingFace)
**Compute:** T4 GPU, <2GB VRAM, ~100ms/frame
**Risk:** Low — well-established, trivial to run

### Stage 2: Reprojection (OUR CONTRIBUTION)
**Input:** Depth map + hand/object 3D positions + camera pose
**Output:** Sparse ego frame + occlusion mask
**Code:** Custom Python (~200 lines, OpenCV + NumPy)
**Key:** Place virtual camera at head position, project 3D scene into ego view
**Risk:** Medium — geometry is straightforward but getting "good looking" output requires tuning

### Stage 3: Inpainting
**Input:** Sparse ego frame + occlusion mask + surrounding exo frames
**Output:** Clean ego frame with holes filled
**Model:** ProPainter (ICCV 2023, 6.6K stars)
**Compute:** 3GB VRAM at 320×240
**Risk:** Low — well-tested, HuggingFace demo available

### Stage 4: Video Generation (Optional)
**Input:** Sequence of inpainted ego frames
**Output:** Smooth ego video clip
**Model:** Wan2.2 TI2V-5B (single GPU, HuggingFace)
**Compute:** Colab Pro (A100, ~9 min for 5 sec clip)
**Alternative:** Morphic frames-to-video (Wan2.2 + LoRA for frame interpolation)
**Risk:** High — heavy compute, may not be needed for class project

### Stage 5: Annotation
**Input:** Generated ego frame
**Output:** {"verb", "noun", "caption", "contact"}
**Model:** GPT-5.5 or Florence-2
**Risk:** Low — trivial to run

## Build Order (Step-by-Step)

### Phase 1: Get One Frame Working (Week 1)
1. Pick one Ego-Exo4D clip with ground truth ego
2. Run Depth Anything V2 on exo frame → verify depth looks correct
3. Run FollowMyHold on the clip → verify hand + object reconstruction
4. Implement reprojection (our code) → compare with ground truth ego
5. Run ProPainter to fill holes → evaluate visual quality

### Phase 2: Pipeline Polish (Week 2)
6. Tune reprojection parameters (head position, camera intrinsics)
7. Add annotation (VLM captioning)
8. Generate 10-20 ego frames from different exo views
9. Qualitative evaluation: side-by-side comparisons
10. If time: Wan2.2 video generation for a polished demo

### Phase 3: Paper (Week 3)
11. Write method section
12. Run evaluation metrics
13. Write results + discussion
14. Record demo video

## Temporal Consistency Strategy

We handle temporal consistency at THREE levels:

1. **Hand tracking:** FollowMyHold tracks hands across frames (HaMeR runs per-frame but the geometric guidance enforces consistency)
2. **Inpainting:** ProPainter uses temporal context (flow-guided propagation) so filled regions are consistent across frames
3. **Video generation:** Wan2.2 is inherently temporally consistent (video diffusion model)

Physics accuracy comes from FollowMyHold's geometric guidance optimization, which enforces that hand-object contacts are physically plausible.

## Compute Requirements (Colab Pro)

| Stage | GPU | VRAM | Time/Frame |
|-------|-----|------|------------|
| FollowMyHold | V100/A100 | ~16GB | ~30s |
| Depth Anything V2 | T4 | ~2GB | ~0.1s |
| Reprojection | CPU | 0 | ~0.01s |
| ProPainter | T4 | ~3GB | ~0.05s |
| Wan2.2 TI2V-5B | A100 | ~20GB | ~9 min/5s clip |
| Annotation | CPU | 0 | ~1s |

**Total per ego frame (without video):** ~30 seconds  
**Total with video:** ~9 minutes for a 5-second clip
