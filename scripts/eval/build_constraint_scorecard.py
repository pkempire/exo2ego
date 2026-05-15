"""Aggregate per-frame egojudge_constraints.py outputs into a scorecard table.

Reads all experiments/<take>/runs/exo_*/judge/constraints.json files and emits:
  - <out>/scorecard.json
  - <out>/scorecard.md     (markdown table for paper / docs)
  - <out>/scorecard.tex    (LaTeX booktabs table for report.tex)

Usage:
    python scripts/build_constraint_scorecard.py \
        --runs experiments/new_upload_sync/take2/runs \
        --out  experiments/new_upload_sync/take2/eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="take2")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for j in sorted(runs_dir.glob("exo_*/judge/constraints.json")):
        d = json.loads(j.read_text())
        s = d["judge_result"]["summary"]
        frame = j.parents[1].name
        rows.append({
            "frame": frame,
            "n_constraints": len(d["constraints_used"]),
            "n_pass": s.get("n_pass"),
            "n_fail": s.get("n_fail"),
            "n_uncertain": s.get("n_uncertain"),
            "score": float(s.get("score_weighted") or 0.0),
            "verdict": s.get("verdict"),
            "top_failures": (s.get("top_failures") or [])[:3],
        })

    summary = {
        "label": args.label,
        "n_frames": len(rows),
        "mean_score": sum(r["score"] for r in rows) / max(1, len(rows)),
        "n_total_pass": sum(r["n_pass"] for r in rows),
        "n_total_fail": sum(r["n_fail"] for r in rows),
        "rows": rows,
    }
    (out_dir / "scorecard.json").write_text(json.dumps(summary, indent=2))

    md = [f"# Constraint scorecard — {args.label}", ""]
    md += [f"`n_frames={len(rows)}`  ·  `mean_score={summary['mean_score']:.3f}`  ·  "
           f"`total pass/fail={summary['n_total_pass']}/{summary['n_total_fail']}`", ""]
    md += ["| frame | constraints | pass | fail | uncertain | score | verdict | top failures |",
           "|-------|------------:|-----:|-----:|----------:|------:|---------|--------------|"]
    for r in rows:
        fails = "; ".join(r["top_failures"]) if r["top_failures"] else "—"
        md.append(f"| {r['frame']} | {r['n_constraints']} | {r['n_pass']} | "
                  f"{r['n_fail']} | {r['n_uncertain']} | {r['score']:.2f} | "
                  f"{r['verdict']} | {fails} |")
    (out_dir / "scorecard.md").write_text("\n".join(md) + "\n")

    tex = [r"\begin{table}[H]",
           r"\centering",
           r"\small",
           r"\caption{Per-frame LLM constraint-judge results on \texttt{" + args.label + r"} (gpt-5.1).",
           r"Score is the always-on-weighted pass fraction; a verdict of \emph{usable} requires score $\ge 0.6$.}",
           r"\label{tab:constraint-scorecard-" + args.label + r"}",
           r"\begin{tabular}{lrrrrrr}",
           r"\toprule",
           r"Frame & \#C & Pass & Fail & Uncert. & Score & Verdict \\",
           r"\midrule"]
    bsl = "\\\\"  # avoid backslash inside f-string expression part
    for r in rows:
        safe_frame = r["frame"].replace("_", "\\_")
        tex.append(f"{safe_frame} & {r['n_constraints']} & {r['n_pass']} & "
                   f"{r['n_fail']} & {r['n_uncertain']} & {r['score']:.2f} & {r['verdict']} {bsl}")
    mean_cell = f"\\textbf{{{summary['mean_score']:.2f}}}"
    tex += [r"\midrule",
            "\\textbf{Mean} & --- & --- & --- & --- & " + mean_cell + " & --- " + bsl,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"]
    (out_dir / "scorecard.tex").write_text("\n".join(tex) + "\n")

    print(f"wrote {out_dir / 'scorecard.md'}")
    print(f"wrote {out_dir / 'scorecard.tex'}")
    print(f"frames={len(rows)} mean_score={summary['mean_score']:.3f}")


if __name__ == "__main__":
    main()
