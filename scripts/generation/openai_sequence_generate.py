#!/usr/bin/env python3
"""Run OpenAI exo-to-ego image edits on a sampled frame sequence.

Frame 0 uses only the current exo frame. Later frames optionally include the
previous generated ego frame as an additional image so the model can preserve
camera style and appearance across time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-manifest", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--size", default="1536x1024")
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--no-prev-reference", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.frames_manifest).read_text())
    frames = manifest["frames"][args.start_index :]
    if args.limit is not None:
        frames = frames[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(exist_ok=True)

    seq_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frames_manifest": str(Path(args.frames_manifest).resolve()),
        "prompt": str(Path(args.prompt).resolve()),
        "model": args.model,
        "size": args.size,
        "quality": args.quality,
        "use_previous_reference": not args.no_prev_reference,
        "frames": [],
    }

    previous_out = None
    # If resuming from nonzero index, find prior generated frame if present.
    if args.start_index > 0:
        prior = gen_dir / f"ego_{args.start_index - 1:03d}.png"
        if prior.exists():
            previous_out = prior

    for item in frames:
        idx = int(item["index"])
        out = gen_dir / f"ego_{idx:03d}.png"
        cmd = [
            sys.executable,
            "scripts/egojudge_openai.py",
            "--image",
            item["path"],
            "--prompt",
            args.prompt,
            "--out",
            str(out),
            "--model",
            args.model,
            "--size",
            args.size,
            "--quality",
            args.quality,
        ]
        refs = []
        if previous_out is not None and not args.no_prev_reference:
            refs.append(str(previous_out))
            cmd += ["--reference-images", str(previous_out)]
        record = {
            "index": idx,
            "time": item["time"],
            "exo": item["path"],
            "out": str(out.resolve()),
            "reference_images": refs,
            "status": "dry_run" if args.dry_run else "pending",
        }
        print(" ".join(cmd))
        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True)
                record["status"] = "ok"
                previous_out = out
            except subprocess.CalledProcessError as exc:
                record["status"] = "failed"
                record["returncode"] = exc.returncode
                # Keep going to expose where failures happen, but do not update
                # previous_out with a missing frame.
        seq_manifest["frames"].append(record)
        (out_dir / "sequence_generation_manifest.json").write_text(json.dumps(seq_manifest, indent=2))
        if not args.dry_run:
            time.sleep(args.sleep)
    print(f"Wrote {out_dir / 'sequence_generation_manifest.json'}")


if __name__ == "__main__":
    main()
