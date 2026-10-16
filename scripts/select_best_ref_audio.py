#!/usr/bin/env python3
"""
Select the acoustically and phonetically optimal reference audio clip from holdout data.
Criteria:
1. Duration between 5.0s and 7.5s (sweet spot for GPT-SoVITS prompt conditioning)
2. Complete tone coverage: presence of Tone 1, 2, 3, and 4
3. High voicing continuity and clean F0 pitch contour
4. No digital clipping (peak < 0.95)
"""

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import soundfile as sf
import numpy as np
from pypinyin import pinyin, Style

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tutor" / "backend"))
from audio_utils import extract_f0_contour

def count_distinct_tones(text: str) -> set:
    tones = set()
    pyn = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    for item in pyn:
        w = item[0]
        for digit in "1234":
            if digit in w:
                tones.add(int(digit))
    return tones

def evaluate_clip(item: dict) -> float:
    wav_path = Path(item["wav_path"])
    if not wav_path.exists():
        return -1.0

    try:
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        dur = len(data) / sr

        # 1. Duration score: ideal is 5.5s - 6.5s
        if dur < 4.5 or dur > 8.0:
            return -1.0
        dur_score = 100.0 - abs(dur - 6.0) * 15.0

        # 2. Peak amplitude score
        peak = float(np.max(np.abs(data)))
        if peak >= 0.98:
            return -1.0  # reject clipping risk
        peak_score = 100.0 if (0.5 <= peak <= 0.92) else 70.0

        # 3. Tone variety score (100 points for all 4 tones)
        tones = count_distinct_tones(item["text"])
        tone_score = len(tones) * 25.0

        # 4. F0 pitch contour quality
        f0_info = extract_f0_contour(data, sr=sr, n_points=80)
        if not f0_info["has_voicing"]:
            return -1.0
        voiced_ratio = sum(1 for v in f0_info["raw_f0"] if v is not None) / max(len(f0_info["raw_f0"]), 1)
        voicing_score = min(100.0, voiced_ratio * 140.0)

        # Composite score
        total_score = 0.35 * tone_score + 0.30 * voicing_score + 0.20 * dur_score + 0.15 * peak_score
        return float(total_score)

    except Exception as e:
        print(f"Error evaluating {wav_path}: {e}")
        return -1.0

def main():
    parser = argparse.ArgumentParser(description="Select optimal reference clip from holdout.")
    parser.add_argument("--holdout_dir", default="data/train_v3/csmsc_holdout", help="Path to holdout directory")
    parser.add_argument("--output_wav", default="data/train_v3/csmsc/csmsc_ref.wav", help="Output path for selected ref wav")
    parser.add_argument("--output_txt", default="data/train_v3/csmsc/csmsc_ref.txt", help="Output path for selected ref text")
    args = parser.parse_args()

    holdout_path = (PROJECT_ROOT / args.holdout_dir).resolve() if not Path(args.holdout_dir).is_absolute() else Path(args.holdout_dir)
    manifest_file = holdout_path / "holdout_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Holdout manifest not found: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"Evaluating {len(items)} holdout candidates for optimal prompt reference...")
    scored = []
    for it in items:
        sc = evaluate_clip(it)
        if sc > 0:
            scored.append((sc, it))

    if not scored:
        raise RuntimeError("No suitable candidate found in holdout!")

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_item = scored[0]

    print(f"\nTop Reference Candidate (Score: {best_score:.1f}/100):")
    print(f"  ID      : {best_item['id']}")
    print(f"  Text    : {best_item['text']}")
    print(f"  Duration: {best_item['duration']:.2f}s")
    print(f"  Wav     : {best_item['wav_path']}")

    out_wav = (PROJECT_ROOT / args.output_wav).resolve() if not Path(args.output_wav).is_absolute() else Path(args.output_wav)
    out_txt = (PROJECT_ROOT / args.output_txt).resolve() if not Path(args.output_txt).is_absolute() else Path(args.output_txt)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    data, sr = sf.read(best_item["wav_path"])
    sf.write(str(out_wav), data, sr)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(best_item["text"])

    print(f"\nSaved optimal reference audio to {out_wav}")
    print(f"Saved prompt text to {out_txt}")

if __name__ == "__main__":
    main()
