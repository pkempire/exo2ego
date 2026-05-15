#!/usr/bin/env python3
"""Generate an EgoJudge candidate with OpenAI image editing."""

from __future__ import annotations

import argparse
import base64
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-images", nargs="*", default=None, help="Optional extra images, e.g. previous generated frame for temporal style consistency.")
    parser.add_argument("--model", default=os.environ.get("EGOJUDGE_IMAGE_MODEL", "gpt-image-1"))
    parser.add_argument("--size", default="1536x1024")
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--input-fidelity", default=None)
    parser.add_argument("--metadata-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Put it in .env first.")

    prompt = Path(args.prompt).read_text()
    client = OpenAI()

    out = Path(args.out)
    meta_path = Path(args.metadata_out) if args.metadata_out else out.with_suffix(out.suffix + ".json")
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image": str(Path(args.image).resolve()),
        "prompt": str(Path(args.prompt).resolve()),
        "out": str(out.resolve()),
        "model": args.model,
        "size": args.size,
        "quality": args.quality,
        "input_fidelity": args.input_fidelity,
        "reference_images": [str(Path(p).resolve()) for p in args.reference_images or []],
        "status": "pending",
    }

    open_files = [open(args.image, "rb")]
    for ref in args.reference_images or []:
        open_files.append(open(ref, "rb"))
    try:
        kwargs = {
            "model": args.model,
            "image": open_files if len(open_files) > 1 else open_files[0],
            "prompt": prompt,
            "size": args.size,
            "quality": args.quality,
        }
        if args.input_fidelity:
            kwargs["input_fidelity"] = args.input_fidelity
        try:
            response = client.images.edit(**kwargs)
        except TypeError as exc:
            if "input_fidelity" not in kwargs:
                raise
            meta["input_fidelity_retry"] = "client rejected input_fidelity; retried without it"
            kwargs.pop("input_fidelity", None)
            response = client.images.edit(**kwargs)
        except OpenAIError as exc:
            meta["status"] = "failed"
            meta["error_type"] = exc.__class__.__name__
            meta["error"] = str(exc)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"OpenAI image generation failed; wrote {meta_path}")
            raise
    finally:
        for f in open_files:
            f.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    b64 = response.data[0].b64_json
    out.write_bytes(base64.b64decode(b64))
    meta["status"] = "ok"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved {out}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
