# Exo2Ego — Paper Outline

## Title
**Exo2Ego: A Modular Perception Pipeline for Converting Third-Person Video to Egocentric Robot Training Data**

## Abstract (draft)
Training Vision-Language-Action (VLA) models for robotic manipulation requires vast quantities of egocentric (first-person) data, which is expensive and time-consuming to collect. Meanwhile, third-person video of human manipulation tasks is abundant on the internet and in industrial settings. We present Exo2Ego, a modular perception pipeline that converts third-person video into robot-ready egocentric frames with structured annotations. Our approach composes state-of-the-art pretrained models — depth estimation, segmentation, hand/object pose estimation, and geometric reprojection — into a coherent pipeline without requiring end-to-end training. We demonstrate the pipeline on indoor tabletop assembly tasks and evaluate the utility of the generated data for VLA pre-training. Our results show that geometrically-grounded reprojection with human-comprehensible features produces useful training signals while avoiding the hallucination problems of purely generative approaches.

## 1. Introduction
- VLA models (π0, π0.5, RT-2) need diverse egocentric training data
- Data collection bottleneck: PI uses ~100 teleoperators 24/7
- Third-person video is abundant but unused due to viewpoint gap  
- Our solution: modular perception pipeline for exo→ego conversion
- Key insight: human-comprehensible features + physical constraints > end-to-end hallucination

## 2. Related Work

### 2.1 VLA Models and the Data Bottleneck
- π0 (Physical Intelligence, 2025): flow matching VLA, 7 robot platforms
- π0.5 (Physical Intelligence, 2025): co-training on heterogeneous data
- Emergent human-to-robot transfer (PI, 2025): ego human video improves policies 2x
- PhysBrain (PI, 2025): egocentric→embodiment translation pipeline

### 2.2 Egocentric Vision Datasets
- Ego4D (Grauman et al., 2022): 3,800 hours FPV
- Ego-Exo4D (Grauman et al., 2024): paired ego+exo, largest such dataset
- Epic-Kitchens (Damen et al., 2018): ego cooking

### 2.3 Cross-View Synthesis
- Exo2Ego-V (Liu et al., NeurIPS 2024): diffusion-based exo→ego video
- Luo et al. (2024): two-stage pipeline (layout → diffusion detail)
- Cross-View Video Synthesis (2021): early GAN-based approach

### 2.4 Composing Pretrained Perception
- SceneComplete (Agarwal et al., 2024): VLM + segmentation + inpainting + image-to-3D
- VideoMimic (UC Berkeley, 2025): real-to-sim from internet video
- Our approach is closer to SceneComplete's philosophy than end-to-end diffusion

### 2.5 Manocha Lab Context
- CSCPR (Liang & Manocha, 2025): cross-source RGB-D place recognition
- ProNav/AMCO: multimodal sensor fusion for robot perception
- VLM-Social-Nav: VLMs for robot navigation

## 3. Method

### 3.1 Problem Formulation
Given: third-person video V_exo of a manipulation task
Output: egocentric frames F_ego + structured annotations A

### 3.2 Pipeline Overview
(Architecture diagram)

### 3.3 Stage 1: Scene Geometry
- Depth estimation (Depth Anything v2)
- Camera pose estimation (COLMAP for moving, fixed for static)

### 3.4 Stage 2: Segmentation
- SAM 3 for class-agnostic masks
- Hand-specific segmentation via MediaPipe/HaMeR
- Object masks from Grounded-SAM

### 3.5 Stage 3: Pose Estimation
- Hand mesh via HaMeR (MANO model)
- Object 6-DoF via FoundationPose
- Temporal smoothing via EKF

### 3.6 Stage 4: Geometric Reprojection
- Virtual camera placement at actor's head position
- Point cloud projection from depth + segmentation
- Occlusion handling via depth buffering

### 3.7 Stage 5: Diffusion Cleanup (optional)
- Exo2Ego-V for photorealistic frame completion
- Guided by geometric constraints from Stage 4

### 3.8 Stage 6: Annotation Generation
- Output schema definition
- Automated captioning via VLM

## 4. Implementation

### 4.1 Model Selection
(Table of chosen models per stage with justification)

### 4.2 Dataset
- Input: Ego-Exo4D paired data (for validation)
- Input: custom phone videos of assembly tasks
- Output: generated ego frames + annotations

### 4.3 System Details
- Python pipeline, modular wrappers
- Each stage independently runnable and evaluable

## 5. Experiments and Evaluation

### 5.1 Qualitative Evaluation
- Side-by-side: exo input → ego output
- Comparison with end-to-end generative baselines

### 5.2 Geometric Accuracy
- Reprojection error on known objects
- Depth consistency across views

### 5.3 Downstream Utility (stretch goal)
- Train simple policy on generated data
- Compare with policy trained on real ego data only

### 5.4 Ablation Studies
- Contribution of each pipeline stage
- With vs. without diffusion cleanup

## 6. Discussion

### 6.1 Limitations
- Single indoor tabletop domain
- Requires reasonably clean input video
- Dependent on pretrained model quality

### 6.2 Future Work
- Extend to multi-camera factory settings
- Integrate with simulators for policy training
- Real-time version for teleoperation

## 7. Conclusion
Summary of contributions and significance for robot learning.

---

## References
[To be populated with BibTeX entries]

## Appendix
- Full output schema specification
- Model download links and setup instructions
- Sample input/output pairs
