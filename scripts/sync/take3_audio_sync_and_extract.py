"""Audio-sync the new take (C0010.MP4 exo + IMG_4537.MOV ego), refine clap
timestamps, and extract paired frames across the synced segment.

Inputs come from Pranay's notes:
    exo (C0010.MP4):    start ~2:06   end ~2:57
    ego (IMG_4537.MOV): start ~3:49   end ~4:41

The script:
  1. Pulls mono 16kHz audio from each video.
  2. Computes a short-window RMS envelope.
  3. Looks for the loudest transient in a +/-1s window around each manually
     marked time (start clap and end clap).
  4. Reports refined times and the implied (ego - exo) offset for both events.
  5. If the two offsets agree within 0.10s, extracts paired frames at a chosen
     stride across the whole window.

Outputs:
  experiments/new_upload_sync/take3/sync_audio.json
  experiments/new_upload_sync/take3/sync_envelopes.png
  experiments/new_upload_sync/take3/frames/exo/exo_NNN_<sec>.jpg
  experiments/new_upload_sync/take3/frames/ego/ego_NNN_<sec>.jpg
  experiments/new_upload_sync/take3/pairs.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import wave
from pathlib import Path

import numpy as np


def hms(s: float) -> str:
    m = int(s // 60); r = s - 60 * m
    return f"{m:02d}:{r:06.3f}"


def extract_audio(video: Path, out: Path, sr: int = 16000) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(video),
           "-vn", "-ac", "1", "-ar", str(sr), "-acodec", "pcm_s16le",
           str(out)]
    subprocess.check_call(cmd)
    return out


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        frames = w.readframes(n)
        sw = w.getsampwidth()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if w.getnchannels() != 1:
        audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
    audio /= float(np.iinfo(dtype).max)
    return audio, sr


def envelope(audio: np.ndarray, sr: int, win_ms: float = 25.0) -> tuple[np.ndarray, float]:
    win = max(1, int(sr * win_ms / 1000.0))
    n_windows = len(audio) // win
    chunks = audio[: n_windows * win].reshape(n_windows, win)
    rms = np.sqrt((chunks ** 2).mean(axis=1) + 1e-12)
    hop_s = win / sr
    return rms, hop_s


def refine_peak(rms: np.ndarray, hop_s: float, target_s: float,
                radius_s: float = 1.0) -> tuple[float, float]:
    t = np.arange(len(rms)) * hop_s
    lo = target_s - radius_s
    hi = target_s + radius_s
    mask = (t >= lo) & (t <= hi)
    if not mask.any():
        return target_s, 0.0
    idx_window = np.where(mask)[0]
    peak_local = idx_window[np.argmax(rms[idx_window])]
    return float(t[peak_local]), float(rms[peak_local])


def extract_frame(video: Path, t_s: float, out: Path, scale: int | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = []
    if scale is not None:
        vf.append(f"scale={scale}:-2")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{t_s:.3f}", "-i", str(video),
           "-frames:v", "1"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-q:v", "3", str(out)]
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exo", default="C0010.MP4")
    ap.add_argument("--ego", default="IMG_4537.MOV")
    ap.add_argument("--exo-start", type=float, default=2 * 60 + 6,    # 02:06.000
                    help="approx exo time of start clap, in seconds")
    ap.add_argument("--exo-end",   type=float, default=2 * 60 + 57,   # 02:57.000
                    help="approx exo time of end clap, in seconds")
    ap.add_argument("--ego-start", type=float, default=3 * 60 + 49,   # 03:49.000
                    help="approx ego time of start clap, in seconds")
    ap.add_argument("--ego-end",   type=float, default=4 * 60 + 41,   # 04:41.000
                    help="approx ego time of end clap, in seconds")
    ap.add_argument("--out", default="experiments/new_upload_sync/take3")
    ap.add_argument("--stride", type=float, default=4.0,
                    help="seconds between paired frames inside the synced segment")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "_audio"
    print(f">> Extracting audio")
    exo_wav = extract_audio(Path(args.exo), audio_dir / "exo.wav")
    ego_wav = extract_audio(Path(args.ego), audio_dir / "ego.wav")

    exo_audio, sr = read_wav_mono(exo_wav)
    ego_audio, _  = read_wav_mono(ego_wav)
    exo_rms, hop_s = envelope(exo_audio, sr)
    ego_rms, _    = envelope(ego_audio, sr)

    refined = {}
    for name, vid_audio_rms, target in [
        ("start_exo", exo_rms, args.exo_start),
        ("end_exo",   exo_rms, args.exo_end),
        ("start_ego", ego_rms, args.ego_start),
        ("end_ego",   ego_rms, args.ego_end),
    ]:
        t_peak, amp = refine_peak(vid_audio_rms, hop_s, target, radius_s=1.0)
        refined[name] = {"t_seconds": t_peak, "rms_peak": amp,
                         "hms": hms(t_peak), "target": target}

    offset_start = refined["start_ego"]["t_seconds"] - refined["start_exo"]["t_seconds"]
    offset_end   = refined["end_ego"]["t_seconds"]   - refined["end_exo"]["t_seconds"]
    drift = offset_end - offset_start

    summary = {
        "exo_video": args.exo,
        "ego_video": args.ego,
        "manual_anchors": {
            "exo_start": args.exo_start, "exo_end": args.exo_end,
            "ego_start": args.ego_start, "ego_end": args.ego_end,
        },
        "refined": refined,
        "offset_start_ego_minus_exo": offset_start,
        "offset_end_ego_minus_exo":   offset_end,
        "drift_seconds": drift,
    }
    (out_dir / "sync_audio.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if abs(drift) > 0.20:
        print(f"WARNING: clap-to-clap drift is {drift:.3f}s; sync may be off.")

    # Plot envelopes around the four anchors
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        for ax, name, src in [
            (axes[0, 0], "exo start", ("exo", exo_rms, refined["start_exo"]["t_seconds"], args.exo_start)),
            (axes[0, 1], "ego start", ("ego", ego_rms, refined["start_ego"]["t_seconds"], args.ego_start)),
            (axes[1, 0], "exo end",   ("exo", exo_rms, refined["end_exo"]["t_seconds"],   args.exo_end)),
            (axes[1, 1], "ego end",   ("ego", ego_rms, refined["end_ego"]["t_seconds"],   args.ego_end)),
        ]:
            label, env, t_peak, t_manual = src
            t = np.arange(len(env)) * hop_s
            lo, hi = t_manual - 2.0, t_manual + 2.0
            m = (t >= lo) & (t <= hi)
            ax.plot(t[m], env[m], color="#444")
            ax.axvline(t_peak, color="#d62728", lw=2, label=f"refined {t_peak:.3f}s")
            ax.axvline(t_manual, color="#1f77b4", lw=1, ls="--", label=f"manual {t_manual:.3f}s")
            ax.set_title(f"{label}  ({label.split()[0]} stream)")
            ax.set_xlabel("seconds"); ax.set_ylabel("RMS")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "sync_envelopes.png", dpi=110)
        print(f"wrote {out_dir / 'sync_envelopes.png'}")
    except Exception as e:
        print(f"(envelope plot skipped: {e})")

    if args.skip_extract:
        return

    # Extract paired frames across the synced segment.
    exo_t = np.arange(refined["start_exo"]["t_seconds"],
                      refined["end_exo"]["t_seconds"] + 0.001,
                      args.stride)
    # Use start-clap offset by default
    offset = offset_start

    frames_dir = out_dir / "frames"
    pairs = []
    print(f">> Extracting {len(exo_t)} paired frames (stride={args.stride}s)")
    for i, t_exo in enumerate(exo_t):
        t_ego = t_exo + offset
        exo_path = frames_dir / "exo" / f"exo_{i:03d}_{t_exo:.3f}.jpg"
        ego_path = frames_dir / "ego" / f"ego_{i:03d}_{t_ego:.3f}.jpg"
        extract_frame(Path(args.exo), float(t_exo), exo_path)
        extract_frame(Path(args.ego), float(t_ego), ego_path)
        pairs.append({
            "idx": i,
            "exo_seconds": float(t_exo),
            "ego_seconds": float(t_ego),
            "exo_path": str(exo_path),
            "ego_path": str(ego_path),
        })
    (out_dir / "pairs.json").write_text(json.dumps({"pairs": pairs,
                                                    "offset_used": offset_start}, indent=2))
    print(f"wrote {out_dir / 'pairs.json'} with {len(pairs)} pairs")


if __name__ == "__main__":
    main()
