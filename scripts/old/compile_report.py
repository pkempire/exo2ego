#!/usr/bin/env python3
"""Compile the project report with LaTeX if available, otherwise fallback.

The fallback uses scripts/render_report_pdf.py so the report can be rebuilt on
machines without a TeX distribution.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "report.tex"


def run(cmd: list[str], cwd: Path) -> bool:
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd)
    return proc.returncode == 0


def main() -> None:
    if shutil.which("pdflatex"):
        ok = run(["pdflatex", "-interaction=nonstopmode", TEX.name], TEX.parent)
        ok = run(["pdflatex", "-interaction=nonstopmode", TEX.name], TEX.parent) and ok
        if ok:
            print(f"Wrote {TEX.with_suffix('.pdf')}")
            return
        print("pdflatex failed; falling back to ReportLab renderer.")
    elif shutil.which("tectonic"):
        if run(["tectonic", TEX.name], TEX.parent):
            print(f"Wrote {TEX.with_suffix('.pdf')}")
            return
        print("tectonic failed; falling back to ReportLab renderer.")
    else:
        print("No pdflatex or tectonic binary found; falling back to ReportLab renderer.")

    fallback = ROOT / "scripts" / "render_report_pdf.py"
    proc = subprocess.run([sys.executable, str(fallback)], cwd=ROOT)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
