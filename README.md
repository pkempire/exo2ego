# EgoJudge — Exo-to-Ego Generation with a Constraint Judge

> CMSC498E Robotics Final Project, Spring 2026  
> Parth Kocheta · University of Maryland · Prof. Dinesh Manocha

EgoJudge is a generate-then-judge pipeline for converting third-person manipulation frames into candidate egocentric frames. The core idea is simple: generate several plausible first-person views, then reject the ones that violate explicit physical constraints such as object identity, hand-object contact, first-person viewpoint, arm continuity, and left/right layout.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| Sync paired captures | `scripts/sync/take3_audio_sync_and_extract.py` | Refines clap markers and extracts aligned frames |
| Detect exo objects | `scripts/perception/detect_hosted_hf.py` | Grounding-DINO open-vocabulary boxes |
| Build scene graph | `scripts/perception/vlm_scene_graph.py` | Structured objects, workspace, and spatial constraints |
| Generate ego candidate | `scripts/generation/egojudge_openai.py` | Image-edit generation from exo frame + prompt |
| Judge constraints | `scripts/eval/egojudge_constraints.py` | Per-constraint pass/fail/uncertain with visual evidence |
| Orchestrate one frame | `scripts/orchestration/run_pipeline.py` | Runs detection, prompt, generation/eval plumbing, and panels |

## Run One Frame

```bash
python3 scripts/orchestration/run_pipeline.py \
  --exo experiments/new_upload_sync/take3/frames/exo/exo_005_145.325.jpg \
  --ego experiments/new_upload_sync/take3/generated/gen_005.png \
  --prompt experiments/new_upload_sync/take3/auto_prompt_005/prompt_conditioned_vlm.txt \
  --out experiments/new_upload_sync/take3/runs/exo_005
```

Useful flags:

```bash
--skip-judge   # no API judge call
--skip-detect  # reuse existing detections
--skip-viz3d   # skip point-cloud visualization
--force        # recompute existing outputs
```

## Eval Tools

```bash
python3 scripts/eval/egojudge_constraints.py ...
python3 scripts/eval/build_constraint_scorecard.py ...
python3 scripts/eval/temporal_consistency.py ...
python3 scripts/viz/visualize_pointcloud_reprojection.py ...
```

The repo keeps source code, prompt templates, and reproducible eval plumbing in git. Raw videos, generated experiments, model weights, and the writeup PDF/LaTeX are local-only.

## Repo Layout

```text
scripts/
  orchestration/   one-frame runners and panel builders
  perception/      detection, masks, wearer pose, scene graphs
  generation/      image generation and candidate variants
  eval/            constraint judge, metrics, temporal consistency, scorecards
  viz/             point-cloud and contact-sheet visualizations
  sync/            audio sync and visual sync sheets
  datasets/        dataset-specific frame extraction helpers
  baselines/       geometry and mask-conditioned baselines
  old/             superseded report/render helpers kept for reproducibility
prompts/           reusable prompt templates
docs/              public lightweight notes
```

## Setup

```bash
pip install -r requirements.txt
# optional: add OPENAI_API_KEY and HF_TOKEN to .env
```
