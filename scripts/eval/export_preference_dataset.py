#!/usr/bin/env python3
"""Export EgoJudge candidate scores as pairwise preference JSONL.

This creates a lightweight training artifact for a future reward/ranker model
or DPO-style fine-tuning setup. It does not fine-tune an image generator by
itself; it captures which candidate EgoJudge preferred and why.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--scorecard", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame-id", default="unknown")
    parser.add_argument("--include-reject-winners", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorecard = json.loads(Path(args.scorecard).read_text())
    candidates = scorecard.get("candidates", [])
    prompt_dir = Path(args.prompt_dir)
    prompt_files = {p.stem: str(p.resolve()) for p in prompt_dir.glob("*.txt")}

    records = []
    for a, b in combinations(candidates, 2):
        if a["final_score"] == b["final_score"]:
            continue
        winner, loser = (a, b) if a["final_score"] > b["final_score"] else (b, a)
        if not args.include_reject_winners and str(winner.get("verdict", "")).startswith("reject"):
            continue
        records.append(
            {
                "frame_id": args.frame_id,
                "source_image": str(Path(args.source_image).resolve()),
                "prompt_dir": str(prompt_dir.resolve()),
                "prompt_files": prompt_files,
                "chosen": {
                    "name": winner["name"],
                    "path": winner["path"],
                    "final_score": winner["final_score"],
                    "paired_score": winner.get("paired_score"),
                    "verdict": winner["verdict"],
                },
                "rejected": {
                    "name": loser["name"],
                    "path": loser["path"],
                    "final_score": loser["final_score"],
                    "paired_score": loser.get("paired_score"),
                    "verdict": loser["verdict"],
                },
                "judge_evidence": {
                    "chosen_hands": winner["hand_count_reason"],
                    "chosen_depth": winner["depth_sanity_reason"],
                    "rejected_hands": loser["hand_count_reason"],
                    "rejected_depth": loser["depth_sanity_reason"],
                    "camera": winner["camera_overlap_reason"],
                },
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(record) for record in records) + ("\n" if records else ""))
    print(f"Wrote {len(records)} preference pairs to {out}")


if __name__ == "__main__":
    main()
