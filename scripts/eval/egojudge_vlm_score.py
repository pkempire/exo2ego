#!/usr/bin/env python3
"""Ask an OpenAI vision model to judge ego-view candidates against the source."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gpt-5.1")
    return parser.parse_args()


def data_url(path: str) -> str:
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


def main() -> None:
    args = parse_args()
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Put it in .env first.")

    client = OpenAI()
    content = [
        {
            "type": "input_text",
            "text": (
                "You are EgoJudge. Compare generated egocentric POV candidates against the source exocentric image. "
                "Score each candidate from 0-100 for: (1) ego-view plausibility, (2) object identity preservation, "
                "(3) left/right layout preservation, (4) hand-object interaction plausibility, (5) photorealism. "
                "Penalize debug panels, cropped charts, duplicated hands, missing carton, missing mug, and layout flips. "
                "Return strict JSON with keys candidates: [{name, score, strengths, failures, verdict}]."
            ),
        },
        {"type": "input_text", "text": "SOURCE EXOCENTRIC IMAGE:"},
        {"type": "input_image", "image_url": data_url(args.source)},
    ]

    for cand in args.candidates:
        path = Path(cand)
        if not path.exists():
            continue
        content.append({"type": "input_text", "text": f"CANDIDATE {path.name}:"})
        content.append({"type": "input_image", "image_url": data_url(str(path))})

    response = client.responses.create(
        model=args.model,
        input=[{"role": "user", "content": content}],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(response.output_text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
