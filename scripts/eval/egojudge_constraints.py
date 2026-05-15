"""LLM constraint judge for generated ego frames.

This is the missing eval piece: for each named physical constraint that we put
into the generation prompt (or extracted into the scene graph), ask a VLM
whether the generated ego image actually satisfies it. The output is a
per-constraint pass/fail with quoted evidence, plus an overall score.

The constraints we check come from three sources (in priority order):
  1. The generation prompt's "Hard constraints:" block, if present.
  2. The scene graph's `spatial_constraints` list.
  3. A small set of always-on physical/semantic checks (no extra hands, no
     impossible camera angle, hands attached to forearms, etc).

The judge takes (a) the source exo frame for grounding identity, (b) the
generated ego candidate, and (c) the constraint list, and returns strict JSON.

Usage:
    python scripts/egojudge_constraints.py \
        --exo experiments/new_upload_sync/take2/frames_full/exo/exo_009_130.324.jpg \
        --ego experiments/new_upload_sync/take2/generated/gen_009_exo_130.324.png \
        --prompt experiments/new_upload_sync/take2/variant_prompts/action_phase_strict.txt \
        --out  experiments/new_upload_sync/take2/eval/constraints_009.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------- constraint extraction ----------------------------

ALWAYS_ON = [
    "Hands must be attached to continuous forearms entering from the bottom or sides of the frame.",
    "No extra hands beyond what the exo frame shows.",
    "No additional people visible in the generated ego view.",
    "The camera viewpoint must be plausibly first-person (head-mounted), not overhead or third-person.",
    "Lighting and color cast must be plausible for the same room/scene as the exo frame.",
]


def extract_constraints_from_prompt(prompt_text: str) -> list[str]:
    """Pull bullet items from a 'Hard constraints:' block in the prompt."""
    if not prompt_text:
        return []
    # find a 'Hard constraints:' block (case-insensitive)
    m = re.search(r"(?is)hard constraints:\s*\n(.+?)(?:\n\s*\n|\Z)", prompt_text)
    if not m:
        return []
    block = m.group(1)
    items = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith(("-", "*", "•")):
            items.append(line.lstrip("-*•").strip())
    return items


"""Background objects that are typically OUT of an ego POV looking at a table.
We don't want to require them to be present in the ego image, and we don't
want them in spatial-relation pairs."""
EGO_OUT_OF_FRAME = {"tv", "television", "window", "couch", "sofa", "wall",
                    "ceiling", "floor", "chair", "person", "people"}


def extract_constraints_from_scene_graph(sg: dict) -> list[str]:
    """Use the scene-graph's spatial_constraints + named object pairs.

    Only manipulated/table-surface objects get presence + relation constraints.
    Background objects that the ego camera looking down at the table is
    unlikely to see (tv, window, couch) are excluded — checking them inflates
    the failure count with relations that never could have transferred.
    """
    constraints = list(sg.get("spatial_constraints") or [])
    objs = sg.get("objects") or []
    ws = (sg.get("workspace") or {}).get("name")
    if ws:
        constraints.append(f"The workspace surface ({ws}) must be present.")

    # Dedupe by exact-lowercased name and skip background.
    seen = set()
    table_objs = []
    for o in objs:
        n = (o.get("name") or "").strip().lower()
        if not n or n in seen:
            continue
        if n in EGO_OUT_OF_FRAME:
            continue
        seen.add(n)
        b = o.get("bbox_norm") or []
        if len(b) == 4:
            table_objs.append((n, b[0] + b[2] / 2.0, b[1] + b[3] / 2.0))
        constraints.append(f"The {n} must be present.")

    # Left/right relations only among manipulated table objects.
    for i in range(len(table_objs)):
        for j in range(i + 1, len(table_objs)):
            n1, x1, _ = table_objs[i]
            n2, x2, _ = table_objs[j]
            if n1 == n2:
                continue
            if abs(x1 - x2) > 0.15:
                left, right = (n1, n2) if x1 < x2 else (n2, n1)
                constraints.append(f"{left} must be to the left of {right}.")
    return constraints


def assemble_constraints(prompt_path: str | None, scene_graph_path: str | None,
                         extra: list[str] | None = None) -> list[dict]:
    """Build a deduped, labeled constraint list."""
    src = []
    if prompt_path and Path(prompt_path).exists():
        for c in extract_constraints_from_prompt(Path(prompt_path).read_text()):
            src.append({"source": "prompt", "constraint": c})
    if scene_graph_path and Path(scene_graph_path).exists():
        sg = json.loads(Path(scene_graph_path).read_text())
        for c in extract_constraints_from_scene_graph(sg):
            src.append({"source": "scene_graph", "constraint": c})
    for c in (extra or []):
        src.append({"source": "extra", "constraint": c})
    for c in ALWAYS_ON:
        src.append({"source": "always_on", "constraint": c})
    # dedupe by case-folded text
    seen = set()
    out = []
    for entry in src:
        key = entry["constraint"].lower().strip().rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


# ---------------------------- LLM call ----------------------------

def data_url(path: str) -> str:
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


JUDGE_INSTRUCTIONS = """\
You are EgoJudge-Constraints. You will be shown the exocentric source frame and
one generated egocentric candidate, plus a numbered list of physical/semantic
constraints. Decide for each constraint whether the generated ego image
satisfies it.

Important rules:
- Treat the exo frame as ground truth for object identity, room, and lighting.
- Treat the ego image as the candidate under review.
- A constraint passes only if the ego image clearly satisfies it. If you cannot
  tell from the image, mark "uncertain" — do not invent evidence.
- Quote a 1–2 sentence visual observation from the ego image as evidence for
  each verdict.
- The hand/forearm continuity constraint is hard: if a hand appears detached or
  floats, mark that constraint as fail.
- Spatial relations ("X is to the left of Y") must be evaluated in the
  generated ego image, not in the exo source.

Return strict JSON with exactly this schema:

{
  "candidate": "<filename>",
  "per_constraint": [
    {
      "id": <int>,
      "source": "prompt|scene_graph|always_on|extra",
      "constraint": "<verbatim>",
      "verdict": "pass|fail|uncertain",
      "evidence": "<short observation about the ego image>"
    }
  ],
  "summary": {
    "n_total": <int>,
    "n_pass": <int>,
    "n_fail": <int>,
    "n_uncertain": <int>,
    "score_pass_fraction": <float in [0,1]>,
    "score_weighted": <float in [0,1] where always_on fails are penalized 2x>,
    "verdict": "usable|borderline|reject",
    "top_failures": ["<constraint>", ...]
  }
}
No prose outside the JSON object.
"""


def run_judge(exo_path: str, ego_path: str, constraints: list[dict],
              model: str = "gpt-5.1", max_retries: int = 2) -> dict:
    client = OpenAI()
    constraint_text = "\n".join(
        f"{i+1}. [{c['source']}] {c['constraint']}"
        for i, c in enumerate(constraints)
    )
    content = [
        {"type": "input_text", "text": JUDGE_INSTRUCTIONS},
        {"type": "input_text", "text": f"FILENAME: {Path(ego_path).name}"},
        {"type": "input_text", "text": f"CONSTRAINTS (numbered):\n{constraint_text}"},
        {"type": "input_text", "text": "SOURCE EXOCENTRIC FRAME:"},
        {"type": "input_image", "image_url": data_url(exo_path)},
        {"type": "input_text", "text": "GENERATED EGOCENTRIC CANDIDATE:"},
        {"type": "input_image", "image_url": data_url(ego_path)},
    ]
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
            )
            text = response.output_text.strip()
            # strip code fences if present
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
            return json.loads(text)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"judge failed after {max_retries+1} attempts: {last_err}")


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exo", required=True)
    ap.add_argument("--ego", required=True)
    ap.add_argument("--prompt", default=None, help="generation prompt with Hard constraints block")
    ap.add_argument("--scene-graph", default=None, help="scene_graph.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--extra", nargs="*", default=[], help="extra constraints to append")
    args = ap.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")

    constraints = assemble_constraints(args.prompt, args.scene_graph, args.extra)
    if not constraints:
        print("no constraints assembled — falling back to always_on only", file=sys.stderr)

    result = run_judge(args.exo, args.ego, constraints, model=args.model)
    out = {
        "exo": args.exo,
        "ego": args.ego,
        "prompt": args.prompt,
        "scene_graph": args.scene_graph,
        "constraints_used": constraints,
        "judge_model": args.model,
        "judge_result": result,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    s = result.get("summary", {})
    print(f"wrote {args.out}")
    print(f"  pass={s.get('n_pass')} fail={s.get('n_fail')} uncertain={s.get('n_uncertain')} "
          f"score={s.get('score_weighted')} verdict={s.get('verdict')}")
    for f in (s.get("top_failures") or [])[:5]:
        print(f"  FAIL: {f}")


if __name__ == "__main__":
    main()
