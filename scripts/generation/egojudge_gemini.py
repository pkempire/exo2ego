#!/usr/bin/env python3
"""Generate an EgoJudge candidate with Gemini / Nano Banana."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-image-preview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is missing. Put it in .env first.")

    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit("Install google-genai first: pip install google-genai") from exc

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    image = Image.open(args.image)
    prompt = Path(args.prompt).read_text()

    response = client.models.generate_content(
        model=args.model,
        contents=[image, prompt],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for part in response.parts:
        if getattr(part, "inline_data", None) is not None:
            part.as_image().save(out)
            print(f"Saved {out}")
            return
    raise SystemExit("Gemini returned no image parts.")


if __name__ == "__main__":
    main()
