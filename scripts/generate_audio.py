#!/usr/bin/env python3
"""
Batch Chinese TTS Audio Generator using GPT-SoVITS API.

This script reads Chinese text files (vocab lists, dialogues) and generates
audio files using a running GPT-SoVITS API server.

Prerequisites:
    1. GPT-SoVITS API server must be running:
       python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml

    2. A trained model must be loaded (or use pretrained base models)

Usage:
    python scripts/generate_audio.py --input data/vocab/lesson01.txt --output output/vocab/lesson01
    python scripts/generate_audio.py --input data/dialogues/lesson01.txt --output output/dialogues/lesson01
    python scripts/generate_audio.py --input data/vocab/ --output output/vocab/  # Process entire folder
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


# === Configuration ===
API_URL = "http://127.0.0.1:9880"
DEFAULT_REF_AUDIO = "data/reference_audio/A4_0.wav"
DEFAULT_REF_TEXT = "绿是阳春烟景大块文章的底色四月的林峦更是绿得鲜活秀媚诗意盎然"
DEFAULT_LANG = "zh"


def check_api_running():
    """Check if the GPT-SoVITS API server is running."""
    try:
        resp = requests.get(f"{API_URL}/", timeout=5)
        return True
    except requests.ConnectionError:
        return False


def generate_single(text, output_path, ref_audio=None, ref_text=None,
                    text_lang="zh", speed=1.0):
    """Generate audio for a single text string.

    Args:
        text: Chinese text to synthesize
        output_path: Path to save the output WAV file
        ref_audio: Path to reference audio clip
        ref_text: Transcript of the reference audio
        text_lang: Language code ('zh', 'en', 'ja')
        speed: Speech speed multiplier (0.5-2.0)

    Returns:
        True if successful, False otherwise
    """
    ref_audio = ref_audio or DEFAULT_REF_AUDIO
    ref_text = ref_text or DEFAULT_REF_TEXT

    payload = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": os.path.abspath(ref_audio),
        "prompt_text": ref_text,
        "prompt_lang": "zh",
        "speed_factor": speed,
    }

    try:
        response = requests.post(f"{API_URL}/tts", json=payload, timeout=120)
        if response.status_code == 200 and len(response.content) > 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"  ERROR: API returned status {response.status_code}")
            return False
    except requests.Timeout:
        print(f"  ERROR: Request timed out for: {text[:30]}...")
        return False
    except requests.ConnectionError:
        print(f"  ERROR: Cannot connect to API at {API_URL}")
        return False


def parse_text_file(filepath):
    """Parse a text file into a list of (label, text) tuples.

    Supports two formats:
        1. Simple: one Chinese text per line
           你好
           谢谢
           再见

        2. Labeled: label|text per line (for vocab with pinyin)
           你好|nǐhǎo|hello
           谢谢|xièxie|thanks

    Returns:
        List of (label, chinese_text) tuples
    """
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            chinese_text = parts[0].strip()
            label = f"{i:03d}"

            # If labeled format, use first column as text
            if len(parts) >= 2:
                label = f"{i:03d}_{chinese_text}"

            entries.append((label, chinese_text))

    return entries


def process_file(input_path, output_dir, ref_audio=None, ref_text=None,
                 speed=1.0, dry_run=False):
    """Process a single text file and generate audio for each line.

    Args:
        input_path: Path to input text file
        output_dir: Directory to save output WAV files
        ref_audio: Path to reference audio
        ref_text: Transcript of reference audio
        speed: Speech speed
        dry_run: If True, only print what would be generated
    """
    entries = parse_text_file(input_path)
    filename = Path(input_path).stem

    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Entries:    {len(entries)}")
    print(f"{'='*60}")

    success = 0
    failed = 0

    for i, (label, text) in enumerate(entries):
        output_path = os.path.join(output_dir, f"{filename}_{label}.wav")

        if os.path.exists(output_path):
            print(f"  [{i+1}/{len(entries)}] SKIP (exists): {text[:30]}")
            success += 1
            continue

        if dry_run:
            print(f"  [{i+1}/{len(entries)}] DRY RUN: {text[:30]} → {output_path}")
            continue

        print(f"  [{i+1}/{len(entries)}] Generating: {text[:30]}...", end=" ", flush=True)
        start_time = time.time()

        if generate_single(text, output_path, ref_audio, ref_text, speed=speed):
            elapsed = time.time() - start_time
            print(f"OK ({elapsed:.1f}s)")
            success += 1
        else:
            print(f"FAILED")
            failed += 1

        # Small delay between requests to avoid overloading
        time.sleep(0.5)

    print(f"\nResults: {success} success, {failed} failed, {len(entries)} total")
    return success, failed


def process_directory(input_dir, output_dir, **kwargs):
    """Process all .txt files in a directory."""
    txt_files = sorted(Path(input_dir).glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    total_success = 0
    total_failed = 0

    for txt_file in txt_files:
        sub_output = os.path.join(output_dir, txt_file.stem)
        s, f = process_file(str(txt_file), sub_output, **kwargs)
        total_success += s
        total_failed += f

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_success} success, {total_failed} failed")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Chinese TTS audio from text files using GPT-SoVITS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate audio for a single vocab file
  python scripts/generate_audio.py --input data/vocab/lesson01.txt --output output/vocab/lesson01

  # Generate audio for all dialogue files
  python scripts/generate_audio.py --input data/dialogues/ --output output/dialogues/

  # Generate with slower speed
  python scripts/generate_audio.py --input data/vocab/lesson01.txt --output output/vocab/lesson01 --speed 0.8

  # Dry run (preview without generating)
  python scripts/generate_audio.py --input data/vocab/lesson01.txt --output output/vocab/lesson01 --dry-run

  # Use a different reference voice
  python scripts/generate_audio.py --input data/vocab/lesson01.txt --output output/vocab/lesson01 \\
      --ref-audio data/reference_audio/A4_10.wav \\
      --ref-text "炮眼打好了炸药怎么装"
        """,
    )

    parser.add_argument("--input", "-i", required=True,
                        help="Input text file or directory of .txt files")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for generated WAV files")
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO,
                        help=f"Reference audio clip (default: {DEFAULT_REF_AUDIO})")
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT,
                        help="Transcript of reference audio")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed multiplier, 0.5=slow, 1.0=normal, 2.0=fast (default: 1.0)")
    parser.add_argument("--api-url", default=API_URL,
                        help=f"GPT-SoVITS API URL (default: {API_URL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be generated without actually generating")

    args = parser.parse_args()

    global API_URL
    API_URL = args.api_url

    # Check API
    if not args.dry_run:
        if not check_api_running():
            print(f"ERROR: GPT-SoVITS API is not running at {API_URL}")
            print(f"Start it with:")
            print(f"  cd GPT-SoVITS")
            print(f"  conda activate GPTSoVits")
            print(f"  python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml")
            sys.exit(1)

    input_path = args.input
    if os.path.isdir(input_path):
        process_directory(input_path, args.output,
                          ref_audio=args.ref_audio, ref_text=args.ref_text,
                          speed=args.speed, dry_run=args.dry_run)
    elif os.path.isfile(input_path):
        process_file(input_path, args.output,
                     ref_audio=args.ref_audio, ref_text=args.ref_text,
                     speed=args.speed, dry_run=args.dry_run)
    else:
        print(f"ERROR: {input_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
