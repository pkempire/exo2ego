#!/usr/bin/env python3
"""Final take-level EgoJudge summary runner.

This script is intentionally boring: it does not delete artifacts or hide
intermediate evidence. It gathers the synced frames, flipped ego GT, point-cloud
diagnostic, generated variants, paired sanity metrics, and VLM constraint-judge
outputs into one report-friendly summary.

Default usage is offline/reproducible and only reads existing artifacts:

    python3 scripts/run_final_egojudge_eval.py

If you want to re-run the expensive LLM constraint judge for frame 009 variants,
pass --run-llm-constraints with OPENAI_API_KEY set.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


EXPLICIT_LAYOUT_CONSTRAINTS = [
    "The ghee jar must be to the right of the white salt container, matching the source exo table layout.",
    "The black/orange book should remain on the far/right side of the table, not disappear.",
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def judge_summary(path: Path) -> dict[str, Any] | None:
    data = load_json(path)
    if not data:
        return None
    s = ((data.get("judge_result") or {}).get("summary") or {})
    return {
        "path": rel(path),
        "candidate": Path(data.get("ego", "")).name,
        "pass": s.get("n_pass"),
        "fail": s.get("n_fail"),
        "uncertain": s.get("n_uncertain"),
        "score": s.get("score_weighted"),
        "verdict": s.get("verdict"),
        "top_failures": s.get("top_failures") or [],
    }


def maybe_run_constraint_judge(
    *,
    take_dir: Path,
    model: str,
    out_name: str,
    ego_path: Path,
    prompt_path: Path,
    exo_path: Path,
) -> None:
    out_path = take_dir / "variants_frame009" / out_name
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "egojudge_constraints.py"),
        "--exo",
        str(exo_path),
        "--ego",
        str(ego_path),
        "--prompt",
        str(prompt_path),
        "--out",
        str(out_path),
        "--model",
        model,
        "--extra",
        *EXPLICIT_LAYOUT_CONSTRAINTS,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def paired_means(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    rows = metrics.get("rows") or metrics.get("frames") or []
    if not rows:
        return metrics.get("mean") or metrics.get("summary")
    keys = ["psnr", "ssim", "edge_f1", "hist_w"]
    out = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if vals:
            out[key] = round(sum(vals) / len(vals), 4)
    return out


def fmt_time(seconds: float | int | None) -> str:
    if seconds is None:
        return "?"
    seconds = float(seconds)
    minutes = int(seconds // 60)
    rem = seconds - 60 * minutes
    return f"{minutes:02d}:{rem:06.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take-dir", default="experiments/new_upload_sync/take2")
    ap.add_argument("--paper-dir", default="paper")
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--run-llm-constraints", action="store_true")
    args = ap.parse_args()

    take_dir = (ROOT / args.take_dir).resolve()
    paper_dir = (ROOT / args.paper_dir).resolve()
    eval_dir = take_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "figures").mkdir(parents=True, exist_ok=True)

    prompt_path = take_dir / "variant_prompts" / "action_phase_strict.txt"
    exo_009 = take_dir / "frames_full" / "exo" / "exo_009_130.324.jpg"
    variants = {
        "base": take_dir / "generated" / "gen_009_exo_130.324.png",
        "action_phase_01": take_dir / "variants_frame009" / "action_phase_01.png",
        "action_phase_02": take_dir / "variants_frame009" / "action_phase_02.png",
    }

    if args.run_llm_constraints:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY missing; omit --run-llm-constraints to use cached judge outputs")
        for name, ego_path in variants.items():
            if ego_path.exists():
                maybe_run_constraint_judge(
                    take_dir=take_dir,
                    model=args.model,
                    out_name=(
                        "constraint_judge_base_explicit_layout.json"
                        if name == "base"
                        else f"constraint_judge_{name}_explicit_layout.json"
                    ),
                    ego_path=ego_path,
                    prompt_path=prompt_path,
                    exo_path=exo_009,
                )

    sync = load_json(take_dir / "sync_config.json") or {}
    action_phases = load_json(take_dir / "action_phases.json") or {}
    metrics = load_json(take_dir / "good_window_paired_metrics.json")
    pointcloud = load_json(take_dir / "pointcloud_reprojection_frame009" / "pointcloud_reprojection_demo.json")
    generic_vlm = load_json(take_dir / "variants_frame009" / "vlm_judge.json") or {}

    explicit_judges = {
        "base": judge_summary(take_dir / "variants_frame009" / "constraint_judge_base_explicit_layout.json"),
        "action_phase_01": judge_summary(take_dir / "variants_frame009" / "constraint_judge_action_phase_01_explicit_layout.json"),
        "action_phase_02": judge_summary(take_dir / "variants_frame009" / "constraint_judge_action_phase_02_explicit_layout.json"),
    }
    selected = None
    usable = [(name, row) for name, row in explicit_judges.items() if row]
    if usable:
        selected = max(usable, key=lambda item: (float(item[1].get("score") or 0.0), item[1].get("fail") == 0))[0]

    summary = {
        "take_dir": rel(take_dir),
        "sync": sync,
        "action_phases": action_phases,
        "figures": {
            "sync_good_pairs": rel(paper_dir / "figures" / "take2_good_sync_pairs.jpg"),
            "generated_good_contact_sheet": rel(paper_dir / "figures" / "take2_generated_good_contact_sheet.jpg"),
            "pointcloud_reprojection": rel(paper_dir / "figures" / "take2_pointcloud_reprojection_demo.png"),
            "variant_sheet": rel(paper_dir / "figures" / "take2_variant_sheet_frame009.jpg"),
            "eval_suite_overview": rel(paper_dir / "figures" / "eval_suite_overview.svg"),
        },
        "paired_sanity_means": paired_means(metrics),
        "pointcloud_reprojection": pointcloud,
        "generic_vlm_scores": generic_vlm,
        "explicit_constraint_judge": explicit_judges,
        "selected_by_explicit_constraints": selected,
        "interpretation": [
            "Pixel metrics are sanity checks only because the real ego GT has different crop, tilt, exposure, and head motion.",
            "The point-cloud reprojection is useful as a failure diagnostic: on frame 009 it only fills roughly one third of the target ego canvas, so it cannot carry the final image alone.",
            "The decisive eval is the explicit constraint judge: it tests the same physical/layout constraints we put into the generation prompt, including ghee-right-of-salt and book persistence.",
            "In the cached run, the generic VLM judge preferred action_phase_01, but the explicit constraint judge selects action_phase_02 because it passes the exact layout checks.",
        ],
    }

    out_json = eval_dir / "final_eval_summary.json"
    out_md = eval_dir / "final_eval_summary.md"
    out_json.write_text(json.dumps(summary, indent=2))

    lines = [
        "# Final EgoJudge Summary — take2",
        "",
        "This is the clean final-run summary. It reads existing artifacts by default and never deletes experiment files.",
        "",
        "## Sync",
        "",
    ]
    if sync:
        markers = sync.get("markers") or {}
        for key in ["start_clap", "good_start", "end_clap"]:
            marker = markers.get(key)
            if marker:
                lines.append(
                    f"- `{key}`: exo `{fmt_time(marker.get('exo_sec'))}`, "
                    f"ego `{fmt_time(marker.get('ego_sec'))}`"
                )
        if sync.get("offset_ego_minus_exo_sec") is not None:
            lines.append(f"- ego minus exo offset: `{sync['offset_ego_minus_exo_sec']:.3f}s`")
        if sync.get("good_eval_window"):
            w = sync["good_eval_window"]
            lines.append(
                f"- good eval window: exo `{fmt_time(w.get('exo_start_sec'))}-{fmt_time(w.get('exo_end_sec'))}`, "
                f"ego `{fmt_time(w.get('ego_start_sec'))}-{fmt_time(w.get('ego_end_sec'))}`"
            )
    else:
        lines.append("- Sync config not found.")
    lines += [
        "",
        "## Point-Cloud Reprojection Diagnostic",
        "",
    ]
    if pointcloud:
        lines.append(
            f"- Frame 009 sparse reprojection fill: `{pointcloud.get('filled_pct'):.1f}%` of the ego canvas."
        )
        lines.append(
            "- Interpretation: good for explaining why pure geometry fails; not enough evidence for a faithful ego frame."
        )
    else:
        lines.append("- Point-cloud diagnostic not found.")
    lines += ["", "## Paired Sanity Metrics", ""]
    if summary["paired_sanity_means"]:
        for k, v in summary["paired_sanity_means"].items():
            lines.append(f"- `{k}` mean: `{v}`")
    else:
        lines.append("- Paired metrics not found.")
    lines += ["", "## Explicit Constraint Judge", ""]
    lines.append("| candidate | pass | fail | uncertain | score | verdict | top failures |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for name, row in explicit_judges.items():
        if not row:
            lines.append(f"| {name} |  |  |  |  | missing |  |")
            continue
        failures = "; ".join(row["top_failures"]) if row["top_failures"] else "-"
        lines.append(
            f"| {name} | {row['pass']} | {row['fail']} | {row['uncertain']} | "
            f"{float(row['score']):.3f} | {row['verdict']} | {failures} |"
        )
    lines += [
        "",
        f"Selected by exact physical constraints: `{selected}`.",
        "",
        "## Report Figures",
        "",
    ]
    for label, path in summary["figures"].items():
        lines.append(f"- `{label}`: `{path}`")
    out_md.write_text("\n".join(lines) + "\n")

    print(f"wrote {rel(out_json)}")
    print(f"wrote {rel(out_md)}")
    if selected:
        print(f"selected_by_explicit_constraints={selected}")


if __name__ == "__main__":
    main()
