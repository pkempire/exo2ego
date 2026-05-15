#!/usr/bin/env python3
"""Thin adapter around the official HaMeR demo.

HaMeR is the right replacement for MediaPipe when we need a real 3D hand mesh,
but it has non-trivial setup requirements: a separate Python 3.10 environment,
ViTPose/detectron2 dependencies, demo checkpoints, and the gated MANO_RIGHT.pkl
asset from the MANO website. This adapter makes that status explicit instead of
silently falling back to 2D landmarks and pretending they are a mesh.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hamer-root", default=str(ROOT / "external" / "hamer"))
    p.add_argument("--python", default=sys.executable, help="Python executable for the HaMeR environment.")
    p.add_argument("--body-detector", default="regnety", choices=["regnety", "vitdet"])
    p.add_argument("--run", action="store_true", help="Run HaMeR if prerequisites are present.")
    return p.parse_args()


def check_prereqs(hamer_root: Path, py: str) -> tuple[bool, list[str], list[str]]:
    missing: list[str] = []
    notes: list[str] = []
    if not (hamer_root / "demo.py").exists():
        missing.append(f"{hamer_root}/demo.py")
    if not (hamer_root / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl").exists():
        missing.append("external/hamer/_DATA/data/mano/MANO_RIGHT.pkl")
        notes.append("MANO_RIGHT.pkl is gated; download it manually from https://mano.is.tue.mpg.de")
    try:
        subprocess.run(
            [py, "-c", "import hamer, torch; import vitpose_model"],
            cwd=hamer_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        missing.append("HaMeR Python dependencies in the selected environment")
        notes.append(f"Import check failed with {exc.__class__.__name__}: {exc}")
    return not missing, missing, notes


def main() -> None:
    args = parse_args()
    hamer_root = Path(args.hamer_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ok, missing, notes = check_prereqs(hamer_root, args.python)
    status = {
        "image": str(Path(args.image).resolve()),
        "hamer_root": str(hamer_root.resolve()),
        "python": args.python,
        "ready": ok,
        "missing": missing,
        "notes": notes,
        "setup_commands": [
            "cd external/hamer",
            "python3.10 -m venv .hamer",
            "source .hamer/bin/activate",
            "pip install torch torchvision",
            "pip install -e .[all]",
            "pip install -v -e third-party/ViTPose",
            "bash fetch_demo_data.sh",
            "download MANO_RIGHT.pkl from https://mano.is.tue.mpg.de and place it at _DATA/data/mano/MANO_RIGHT.pkl",
        ],
    }
    status_path = out / "hamer_status.json"

    if not args.run or not ok:
        status["status"] = "blocked" if not ok else "ready_not_run"
        status_path.write_text(json.dumps(status, indent=2))
        print(f"Wrote {status_path}")
        if not ok:
            print("HaMeR is not ready; see missing prerequisites in the status JSON.")
        return

    with tempfile.TemporaryDirectory() as td:
        img_dir = Path(td) / "images"
        img_dir.mkdir()
        ext = Path(args.image).suffix or ".png"
        copied = img_dir / f"input{ext}"
        shutil.copy2(args.image, copied)
        cmd = [
            args.python,
            "demo.py",
            "--img_folder",
            str(img_dir),
            "--out_folder",
            str(out),
            "--batch_size",
            "1",
            "--body_detector",
            args.body_detector,
            "--save_mesh",
            "--full_frame",
        ]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=hamer_root, check=True)

    status["status"] = "ok"
    status["outputs"] = [str(p) for p in sorted(out.glob("*")) if p.name != status_path.name]
    status_path.write_text(json.dumps(status, indent=2))
    print(f"Wrote {status_path}")


if __name__ == "__main__":
    main()
