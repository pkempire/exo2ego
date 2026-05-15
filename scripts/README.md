# `scripts/` layout

Every script in this folder is a self-contained CLI. The orchestrator stitches the most-used ones together.

```
scripts/
├── orchestration/
│   ├── run_pipeline.py            ← main entry point
│   ├── make_pipeline_panel.py
│   └── run_final_egojudge_eval.py
├── perception/
│   ├── detect_hosted_hf.py        ← Grounding-DINO open-vocab detection
│   ├── vlm_scene_graph.py         ← builds scene_graph.json + prompt
│   ├── refine_vlm_scene_masks.py
│   ├── wearer_pose_extract.py     ← MediaPipe head + body
│   ├── wearer_conditioned_prompt.py
│   ├── grounded_sam2_segment.py
│   ├── hamer_adapter.py           ← (blocked on MANO model setup)
│   └── hand_depth_diagnostic.py
├── generation/
│   ├── egojudge_openai.py         ← gpt-image-1 edit
│   ├── egojudge_gemini.py         ← Nano Banana alt
│   ├── best_of_k_generate.py
│   ├── generate_variants.py
│   ├── openai_sequence_generate.py
│   ├── mask_condition_render.py
│   └── build_prompt_ablations.py
├── eval/
│   ├── egojudge_constraints.py    ← the LLM constraint judge (this revision)
│   ├── egojudge_vlm_score.py
│   ├── egojudge_metrics.py
│   ├── egojudge_score.py
│   ├── egojudge_depth_score.py
│   ├── egojudge_prepare.py
│   ├── egojudge_report.py
│   ├── paired_eval_report.py
│   ├── mask_constraint_score.py
│   ├── prompt_following_diagnostic.py
│   ├── robot_usefulness_score.py
│   ├── robotic_consistency_score.py
│   ├── view_calibration_diagnostic.py
│   ├── build_constraint_scorecard.py
│   ├── select_best_variant.py
│   ├── export_preference_dataset.py
│   └── temporal_consistency.py
├── viz/
│   ├── viz_3d_reprojection.py     ← interactive 3D point-cloud HTML
│   ├── visualize_pointcloud_reprojection.py
│   ├── stitch_frames_video.py
│   └── stitch_pair_video.py
├── sync/
│   ├── take3_audio_sync_and_extract.py
│   └── visual_sync_sheet.py
├── datasets/
│   ├── egoexo4d_full_pipeline.py
│   ├── bike_repair_full_pipeline.py
│   ├── mendeley_conditioning.py
│   ├── mendeley_make_pair_sample.py
│   ├── mendeley_nested_pair_sample.py
│   ├── h2o_make_pair_sample.py
│   └── own_capture_pipeline.py
├── baselines/
│   ├── egoworld_pathway.py        ← sparse-ego-pose geometry path
│   ├── egoworld_adapter.py
│   ├── standard_reproject_baseline.py
│   └── sam2_mask_condition_pipeline.py
└── old/
    └── render_*.py                ← superseded PDF renderers
```

Run a single frame end-to-end:

```bash
python scripts/orchestration/run_pipeline.py \
  --exo <exo.jpg> --ego <gen.png> --prompt <prompt.txt> --out <out_dir>
```
