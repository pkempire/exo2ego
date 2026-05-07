# Exo2Ego — Architecture Document

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXO2EGO PIPELINE                              │
│                                                                  │
│  Input: Third-person video (phone camera, surveillance, etc.)   │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ GEOMETRY │ → │SEGMENTAT.│ → │   POSE   │ → │REPROJECT │    │
│  │ Depth +   │   │ SAM +    │   │ Hand +   │   │ View     │    │
│  │ Camera    │   │ Grounded │   │ Object   │   │ Transform│    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│                                                      │          │
│                                                      ▼          │
│                              ┌──────────┐   ┌──────────────┐    │
│                              │ DIFFUSION │ ← │ ANNOTATION   │    │
│                              │ Cleanup   │   │ Depth, Seg,  │    │
│                              │ (optional)│   │ Pose, Caption│    │
│                              └──────────┘   └──────────────┘    │
│                                                      │          │
│  Output: Egocentric frames + structured annotations             │
│  Format: {rgb, depth, segmentation, hand_pose, object_pose,    │
│           contact, caption, verb, noun, cam_intr, cam_extr}     │
└─────────────────────────────────────────────────────────────────┘
```

## Stage 1: Scene Geometry

**Goal:** Estimate depth map and camera pose for every frame.

**Approach:** Use monocular depth estimation (Depth Anything v2) to get per-pixel depth. For moving cameras, run a lightweight SLAM (COLMAP or DROID-SLAM) to track camera extrinsics. For static surveillance cameras, intrinsics are fixed.

**Key constraint:** We target indoor tabletop scenes where geometry is simple (planar surface, known scale). This simplifies reprojection.

**Candidates:**
- Depth Anything v2 (metric depth, indoor-trained)
- UniDepth (universal depth, handles diverse scenes)
- ZoeDepth (good indoor performance)
- DROID-SLAM (lightweight visual SLAM)

## Stage 2: Segmentation

**Goal:** Isolate hands, objects, and the actor from the scene.

**Approach:** SAM 3 for class-agnostic segmentation. Grounded-SAM if we need class labels. Hand-specific models for precise hand masks (critical for manipulation tasks).

**Candidates:**
- SAM 3 (Segment Anything Model v3)
- Grounded-SAM (SAM + grounding DINO for text-conditioned)
- E2FG (Egocentric 2D Future Gaze — specifically for ego hand-object)
- Hand-specific: MediaPipe Hands, HaMeR

## Stage 3: Pose Estimation

**Goal:** Estimate 3D positions and orientations of hands and manipulated objects.

**Approach:** Hand pose via HaMeR (SOTA hand mesh reconstruction). Object pose via FoundationPose or MegaPose. Track across frames for temporal consistency.

**Candidates:**
- HaMeR (Hand Mesh Recovery — SOTA, MANO model)
- FoundationPose (6-DoF object pose, works on novel objects)
- MegaPose (CAD-based, good for known objects)
- ZooPose (zero-shot 6-DoF)
- PHOSA (human-object interaction reconstruction)

## Stage 4: Reprojection

**Goal:** Transform the estimated scene geometry from the third-person view to a first-person view.

**Approach:** Place a virtual camera at the estimated actor's eye/head position (derived from hand pose + scene geometry). Reproject the 3D scene into this new view. Use depth for occlusion handling. Where occluded, mark pixels as "unknown" for the diffusion stage.

**Key insight:** We don't need photorealistic output from this stage — we need geometrically correct object positions. The diffusion stage handles photorealism.

**Implementation:**
- OpenCV `warpPerspective` / `remap` for the base reprojection
- Custom EKF for smoothing camera trajectories
- Point cloud projection from depth + segmentation

## Stage 5: Diffusion Cleanup (Optional)

**Goal:** Fill in missing pixels and add photorealism to the reprojected frames.

**Approach:** Use a video diffusion model conditioned on the reprojected frame and previous generated frames. The model is guided by the geometric constraints (segments, depth) to avoid hallucinating incorrect object positions.

**Candidates:**
- Exo2Ego-V (NeurIPS 2024) — specifically designed for this task
- Stable Video Diffusion (general video generation)
- VideoMimic's rendering pipeline (geometry-first approach)

**Tradeoff:** Skip this for initial prototype. Reprojected frames with correct geometry may be sufficient for VLA pre-training.

## Stage 6: Annotation

**Goal:** Produce structured, robot-ready annotations alongside frames.

**Output schema** (per frame):
```json
{
  "rgb": "frame_00042.jpg",
  "depth": "depth_00042.png",
  "segmentation": "seg_00042.png",
  "hand_pose": {"left": [[x,y,z]*21], "right": [[x,y,z]*21]},
  "object_pose": {"obj_1": [[R|t]_4x4]},
  "contact": 1,
  "caption": "worker tightens bolt with wrench",
  "verb": "tighten",
  "noun": "bolt"
}
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Modular pipeline, not end-to-end | Debuggable, each stage independently improvable |
| Human-comprehensible features | Physical constraints prevent hallucinations |
| Indoor tabletop only (v1) | Known geometry, simple occlusion, clear use case |
| Depth + segmentation before pose | Segmentation guides pose; depth constrains geometry |
| Diffusion is optional | Geometric correctness may be sufficient for training |
| iPhone videos (known intrinsics) | Simplifies camera model when video source is known |

## Scope Boundaries

**In scope (v1):**
- Indoor tabletop assembly/manipulation tasks
- 1-2 people, known surface plane
- Phone camera input (known intrinsics)
- Output: RGB + depth + segmentation + hand pose

**Out of scope (v1):**
- Outdoor / complex environments
- Multi-camera setups
- Real-time processing
- Full 3D scene reconstruction
- Robot policy training (just produce the data)

## References

1. Grauman et al. "Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives." CVPR 2024.
2. Physical Intelligence. "π0: A Vision-Language-Action Flow Model for General Robot Control." 2025.
3. Physical Intelligence. "π0.5: a Vision-Language-Action Model with Open-World Generalization." 2025.
4. Physical Intelligence. "Emergence of Human to Robot Transfer in VLAs." 2025.
5. Physical Intelligence. "PhysBrain: Human Egocentric Data as a Bridge." 2025.
6. Liu et al. "Exo2Ego-V: Cross-View Video Synthesis with Diffusion Models." NeurIPS 2024.
7. Luo et al. "Put Myself in Your Shoes: Lifting Exocentric to Egocentric." 2024.
8. UC Berkeley. "VideoMimic: Learning Robot Skills from Internet Videos." 2025.
9. Agarwal et al. "SceneComplete: Open-World 3D Scene Completion." 2024.
10. Liang & Manocha. "CSCPR: Cross-Source-Context Indoor RGB-D Place Recognition." RA-L 2025.
