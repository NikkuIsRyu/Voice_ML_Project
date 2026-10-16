#!/usr/bin/env python3
"""
Create Slowed Chinese Audio Batch (60-70% speed) using Rubberband/FFmpeg.

Scales refined GPT-SoVITS audio clips to 65% speed (or custom ratio)
preserving pitch, tone contours, and formant quality.

Usage:
    python scripts/create_slow_audio_batch.py
    python scripts/create_slow_audio_batch.py --speed 0.65 --output-dir output_v2_slow/vocab
"""

import argparse
import concurrent.futures
import io
import os
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def check_ffmpeg_filter(filter_name="rubberband"):
    """Check if ffmpeg supports rubberband or fallback to atempo."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True,
            text=True,
            check=True
        )
        return filter_name in proc.stdout
    except Exception:
        return False


def process_file(src_path: Path, dst_path: Path, speed: float = 0.65, use_rubberband: bool = True) -> bool:
    """Process a single WAV file to change tempo/speed."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if use_rubberband:
        audio_filter = f"rubberband=tempo={speed}"
    else:
        # atempo filter (supports 0.5 to 2.0)
        audio_filter = f"atempo={speed}"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-filter:a", audio_filter,
        "-ar", "32000",
        str(dst_path)
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, check=True)
        return dst_path.exists() and dst_path.stat().st_size > 1000
    except subprocess.CalledProcessError as e:
        print(f"Error processing {src_path.name}: {e.stderr.decode('utf-8', errors='ignore')[:200]}")
        return False


def process_batch(input_dir: Path, output_dir: Path, speed: float = 0.65, max_workers: int = 8):
    """Batch convert all WAV files in input_dir to output_dir."""
    print("=" * 65)
    print(f"CREATING SLOW AUDIO BATCH (Speed: {speed * 100:.0f}%)")
    print(f"Input Directory:  {input_dir.resolve()}")
    print(f"Output Directory: {output_dir.resolve()}")
    print("=" * 65)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        sys.exit(1)

    has_rubberband = check_ffmpeg_filter("rubberband")
    print(f"Audio Processing Engine: {'Rubberband (High-Fidelity)' if has_rubberband else 'FFmpeg atempo (WSOLA)'}")

    # Gather all WAV files
    wav_files = sorted(list(input_dir.rglob("*.wav")))
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(wav_files)} audio files to process.")

    start_time = time.time()
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for src in wav_files:
            rel = src.relative_to(input_dir)
            dst = output_dir / rel
            tasks.append(executor.submit(process_file, src, dst, speed, has_rubberband))

        completed = 0
        failed = 0
        for future in concurrent.futures.as_completed(tasks):
            if future.result():
                completed += 1
            else:
                failed += 1

    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed:.2f}s ({completed} succeeded, {failed} failed).")
    return completed, failed


def main():
    parser = argparse.ArgumentParser(description="Create slowed audio batch for Chinese vocabulary.")
    parser.add_argument("--input-dir", default="output_v2/vocab", help="Source directory with normal speed audio")
    parser.add_argument("--output-dir", default="output_v2_slow/vocab", help="Target directory for slowed audio")
    parser.add_argument("--speed", type=float, default=0.65, help="Speed multiplier (default: 0.65 = 65% speed)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads (default: 8)")

    args = parser.parse_args()
    process_batch(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        speed=args.speed,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
