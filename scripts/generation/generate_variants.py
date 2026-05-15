#!/usr/bin/env python3
"""Generate multiple image variants slowly and record metadata.

This is a thin controlled wrapper around scripts/egojudge_openai.py so we can
generate N candidates without accidentally spamming the API.
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
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="variant")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--size", default="1536x1024")
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--max-n", type=int, default=10, help="Hard safety cap.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n > args.max_n:
        raise SystemExit(f"Refusing to generate {args.n}; max-n is {args.max_n}.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image": str(Path(args.image).resolve()),
        "prompt": str(Path(args.prompt).resolve()),
        "model": args.model,
        "size": args.size,
        "quality": args.quality,
        "requested_n": args.n,
        "variants": [],
    }

    for i in range(1, args.n + 1):
        out = out_dir / f"{args.prefix}_{i:02d}.png"
        cmd = [
            sys.executable,
            "scripts/egojudge_openai.py",
            "--image",
            args.image,
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
        print(" ".join(cmd))
        item = {"index": i, "out": str(out.resolve()), "status": "dry_run" if args.dry_run else "pending"}
        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True)
                item["status"] = "ok"
            except subprocess.CalledProcessError as exc:
                item["status"] = "failed"
                item["returncode"] = exc.returncode
        manifest["variants"].append(item)
        (out_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2))
        if i < args.n and not args.dry_run:
            time.sleep(args.sleep)

    print(f"Wrote {out_dir / 'generation_manifest.json'}")


if __name__ == "__main__":
    main()
