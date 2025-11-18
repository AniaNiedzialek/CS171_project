# scripts/extract_frames.py
import subprocess, os, glob, pathlib, shutil, sys

INP = "data/raw/videos"
OUT = "data/frames"
N = 12
SIZE = 256

def ensure_ffmpeg():
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found. Install it (e.g., brew install ffmpeg).")
        sys.exit(1)

def main():
    ensure_ffmpeg()
    mp4s = glob.glob(f"{INP}/**/*.mp4", recursive=True)
    print(f"Found {len(mp4s)} mp4 files under {INP}")
    if not mp4s:
        return

    for mp4 in mp4s:
        div = pathlib.Path(mp4).parts[-2]
        stem = pathlib.Path(mp4).stem
        outdir = pathlib.Path(OUT) / div / stem
        outdir.mkdir(parents=True, exist_ok=True)

        vf = f"fps={N}/4,scale={SIZE}:{SIZE}:flags=lanczos"
        outpat = str(outdir / "%04d.jpg")
        print(f"→ {mp4}\n   out: {outdir}")

        # Skip if frames already exist
        # if any(outdir.glob("*.jpg")):
        #     print("   (skip) frames already exist")
        #     continue

        # Run ffmpeg and show stderr if it fails
        proc = subprocess.run(
            ["ffmpeg","-y","-i",mp4,"-vf",vf,outpat],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if proc.returncode != 0:
            print("   ERROR running ffmpeg:")
            print(proc.stderr)
            continue

        count = len(list(outdir.glob("*.jpg")))
        print(f"   saved {count} frames")

    print("Done!")

if __name__ == "__main__":
    main()
