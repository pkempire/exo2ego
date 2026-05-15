#!/usr/bin/env python3
"""Score generated ego candidates using GroundingDINO+SAM2 mask constraints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-segments", required=True)
    p.add_argument("--candidate-segments", nargs="+", required=True)
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_mask(path: str | Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def by_role(path: str | Path) -> dict[str, list[dict]]:
    data = json.loads(Path(path).read_text())
    roles: dict[str, list[dict]] = {}
    for s in data.get("segments", []):
        roles.setdefault(s.get("role", s.get("label", "unknown")), []).append(s)
    for v in roles.values():
        v.sort(key=lambda s: (s.get("score", 0.0) + s.get("sam2_score", 0.0)), reverse=True)
    return roles


def centroid(seg: dict) -> tuple[float, float]:
    x0, y0, x1, y1 = seg["mask_bbox_xyxy"]
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def role_presence_score(roles: dict[str, list[dict]], expected_roles: list[str]) -> tuple[float, str]:
    present = [r for r in expected_roles if roles.get(r)]
    return len(present) / max(len(expected_roles), 1), f"present roles={present}; expected={expected_roles}"


def min_mask_distance(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() or not b.any():
        return float("inf")
    if np.logical_and(a, b).any():
        return 0.0
    # Distance from b to nearest a pixel.
    inv_a = (~a).astype(np.uint8)
    dist = cv2.distanceTransform(inv_a, cv2.DIST_L2, 5)
    return float(dist[b].min())


def contact_score(roles: dict[str, list[dict]]) -> tuple[float, str]:
    hands = roles.get("hand", [])
    books = roles.get("black_book_or_device", []) or roles.get("orange_book", [])
    if not hands or not books:
        return 0.0, "missing hand or manipulated book/device mask"
    best = float("inf")
    best_pair = None
    for h in hands[:2]:
        hm = load_mask(h["mask_path"])
        for b in books[:2]:
            bm = load_mask(b["mask_path"])
            d = min_mask_distance(hm, bm)
            if d < best:
                best, best_pair = d, (h.get("label"), b.get("label"))
    score = max(0.0, 1.0 - best / 80.0)
    return score, f"min hand-object mask distance={best:.1f}px for {best_pair}"


def spatial_order_score(roles: dict[str, list[dict]]) -> tuple[float, str]:
    black = (roles.get("black_book_or_device") or [None])[0]
    orange = (roles.get("orange_book") or [None])[0]
    if not black or not orange:
        return 0.5, "cannot compare black/orange book order because one role is missing"
    bx, _ = centroid(black)
    ox, _ = centroid(orange)
    # In source frame, orange-spined book is to the right of the black upright object.
    ok = ox >= bx
    return (1.0 if ok else 0.0), f"black/device center x={bx:.1f}, orange-book center x={ox:.1f}; orange should be to the right"


def ego_layout_score(roles: dict[str, list[dict]], image_h: float) -> tuple[float, str]:
    hands = roles.get("hand", [])
    table = (roles.get("table") or [None])[0]
    vals = []
    reasons = []
    if hands:
        hand_y = [centroid(h)[1] / image_h for h in hands[:2]]
        vals.append(max(0.0, min(1.0, (sum(hand_y) / len(hand_y) - 0.35) / 0.35)))
        reasons.append(f"hand centroid y={','.join(f'{y:.2f}' for y in hand_y)}")
    if table:
        ty = centroid(table)[1] / image_h
        vals.append(max(0.0, min(1.0, (ty - 0.45) / 0.25)))
        reasons.append(f"table centroid y={ty:.2f}")
    if not vals:
        return 0.0, "no layout masks available"
    return float(sum(vals) / len(vals)), "; ".join(reasons)


def main() -> None:
    args = parse_args()
    names = args.names or [Path(p).parent.name for p in args.candidate_segments]
    source_roles = by_role(args.source_segments)
    expected = [r for r in ["hand", "black_book_or_device", "orange_book", "table"] if source_roles.get(r)]
    rows = []
    for name, path in zip(names, args.candidate_segments):
        roles = by_role(path)
        data = json.loads(Path(path).read_text())
        image_h = float(data.get("image_shape_hw", [1024, 1024])[0])
        rp, rp_reason = role_presence_score(roles, expected)
        ct, ct_reason = contact_score(roles)
        so, so_reason = spatial_order_score(roles)
        el, el_reason = ego_layout_score(roles, image_h)
        final = 100.0 * (0.30 * rp + 0.30 * ct + 0.20 * so + 0.20 * el)
        rows.append({
            "name": name,
            "segments": path,
            "role_presence": round(100 * rp, 1),
            "role_presence_reason": rp_reason,
            "contact": round(100 * ct, 1),
            "contact_reason": ct_reason,
            "spatial_order": round(100 * so, 1),
            "spatial_order_reason": so_reason,
            "ego_layout": round(100 * el, 1),
            "ego_layout_reason": el_reason,
            "final_mask_constraint_score": round(final, 1),
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mask_constraint_score.json").write_text(json.dumps({
        "source_segments": args.source_segments,
        "expected_roles": expected,
        "candidates": rows,
    }, indent=2))
    lines = [
        "# Mask Constraint Score",
        "",
        f"Expected roles from source: {expected}",
        "",
        "|candidate|final|roles|contact|order|ego layout|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"|{r['name']}|{r['final_mask_constraint_score']}|{r['role_presence']}|{r['contact']}|{r['spatial_order']}|{r['ego_layout']}|")
    lines.append("\n## Evidence")
    for r in rows:
        lines.extend([
            "",
            f"### {r['name']}",
            f"- Roles: {r['role_presence_reason']}",
            f"- Contact: {r['contact_reason']}",
            f"- Order: {r['spatial_order_reason']}",
            f"- Ego layout: {r['ego_layout_reason']}",
        ])
    (out / "mask_constraint_score.md").write_text("\n".join(lines))
    print(f"Wrote {out / 'mask_constraint_score.json'}")
    print(f"Wrote {out / 'mask_constraint_score.md'}")


if __name__ == "__main__":
    main()
