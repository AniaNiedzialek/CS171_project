# scripts/extract_frames.py
import subprocess
import os
import glob
import pathlib
import shutil
import sys
import argparse
from typing import Dict, List


def ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ERROR: ffmpeg not found. Install it (e.g., brew install ffmpeg).")
        sys.exit(1)
    return ffmpeg


def extract_frames(
    input_dir: str = "data/raw/videos",
    output_dir: str = "data/frames",
    frames_per_4s: int = 12,
    size: int = 512,
    overwrite: bool = False,
    verbose: bool = True,
    ffmpeg_bin: str | None = None,
) -> Dict[str, int]:
    if ffmpeg_bin is None:
        ffmpeg_bin = ensure_ffmpeg()

    mp4s = glob.glob(f"{input_dir}/**/*.mp4", recursive=True)
    if verbose:
        print(f"Found {len(mp4s)} mp4 files under {input_dir}")

    if not mp4s:
        return {}

    saved_counts: Dict[str, int] = {}

    for mp4 in mp4s:
        # assume the directory structure includes a division directory right above mp4
        # e.g., data/raw/videos/novice/1234.mp4 -> division name 'novice'
        try:
            div = pathlib.Path(mp4).parts[-2]
        except Exception:
            div = "unknown"
        stem = pathlib.Path(mp4).stem
        outdir = pathlib.Path(output_dir) / div / stem
        outdir.mkdir(parents=True, exist_ok=True)

        # fps is frames_per_4s / 4 for consistency with the original script
        # Use 'force_original_aspect_ratio=decrease' to handle both portrait and landscape videos
        # Then pad to center the result in a square
        vf = f"fps={frames_per_4s}/4,scale={size}:{size}:force_original_aspect_ratio=decrease:flags=lanczos,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2"

        outpat = str(outdir / "%04d.jpg")
        if verbose:
            print(f"→ {mp4}\n   out: {outdir}")

        # Skip if frames already exist
        if not overwrite and any(outdir.glob("*.jpg")):
            if verbose:
                print(f"   (skip) {outdir} already has frames")
            saved_counts[mp4] = len(list(outdir.glob("*.jpg")))
            continue

        # Add format=yuvj420p to handle non-standard YUV color formats
        vf_with_format = vf + ",format=yuvj420p"
        
        proc = subprocess.run(
            [ffmpeg_bin, "-y", "-i", mp4, "-vf", vf_with_format, outpat],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            if verbose:
                print("   ERROR running ffmpeg:")
                print(proc.stderr)
            saved_counts[mp4] = 0
            continue

        count = len(list(outdir.glob("*.jpg")))
        saved_counts[mp4] = count
        if verbose:
            print(f"   saved {count} frames")

    if verbose:
        print("Done!")
    return saved_counts
