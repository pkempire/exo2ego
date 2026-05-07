# Exo2Ego — Third-Person to Egocentric View Synthesis for Robot Training Data

> CMSC498E Robotics Final Project — Spring 2026  
> Parth Kocheta | Prof. Dinesh Manocha

## Problem

Vision-Language-Action (VLA) models for robotics need egocentric (first-person) training data. Physical Intelligence runs ~100 teleoperators 24/7 to collect this data. Third-person video is abundant (surveillance, YouTube, how-to clips) but can't be used directly due to the viewpoint gap.

## Approach

A modular perception pipeline that converts third-person video into robot-ready egocentric data:

**Exo video → Structure from Motion → Depth → Segmentation → Pose → Reprojection → Diffusion cleanup → Annotated ego frames**

We use **human-comprehensible features** (segments, poses, depth) with physical constraints, not end-to-end hallucination. Each stage uses pretrained SOTA models composed together — we are the glue, not the model trainer.

## Pipeline

| Stage | Component | Candidate Models |
|-------|-----------|-----------------|
| 1. Scene Geometry | Depth estimation + camera pose | Depth Anything v2, UniDepth, COLMAP |
| 2. Segmentation | Object + hand + person masks | SAM 3, Grounded-SAM, E2FG |
| 3. Pose Estimation | Hand + object 6-DoF keypoints | HaMeR, ZooPose, FoundationPose |
| 4. Reprojection | Physically-grounded view transform | Custom (OpenCV + EKFs) |
| 5. Diffusion Cleanup | Photorealistic frame completion | Exo2Ego-V, Stable Video Diffusion |
| 6. Annotation | Auto-label output frames | Depth, seg, keypoints, captions |

## Key References

- **Ego-Exo4D** (Grauman et al., 2024) — largest paired ego+exo dataset
- **π0 / π0.5** (Physical Intelligence, 2025) — VLA models needing ego data
- **Exo2Ego-V** (Liu et al., NeurIPS 2024) — diffusion exo→ego translation
- **VideoMimic** (UC Berkeley, 2025) — real-to-sim-to-real from internet video
- **PhysBrain** (PI, 2025) — egocentric→embodiment translation pipeline
- **SceneComplete** (Agarwal et al., 2024) — composing pretrained perception for manipulation
- **CSCPR** (Liang & Manocha, 2025) — cross-source RGB-D place recognition

## Repo Structure

```
exo2ego/
├── src/
│   ├── perception/    # SAM, depth, pose estimation wrappers
│   ├── geometry/      # Camera model, reprojection, SfM
│   ├── generation/    # Diffusion cleanup, frame synthesis
│   └── eval/          # Metrics, visualization, benchmarks
├── data/
│   ├── inputs/        # Sample third-person videos
│   ├── outputs/       # Generated egocentric frames
│   └── annotations/   # Depth maps, seg masks, keypoints
├── notebooks/         # Exploration and demos
├── paper/             # Final report + outline
├── docs/              # Architecture, research notes
└── tests/             # Pipeline tests
```

## Setup

```bash
pip install -r requirements.txt
# Individual model setup in src/*/README.md
```

## Status

Pre-alpha. Pipeline research and component selection in progress.
