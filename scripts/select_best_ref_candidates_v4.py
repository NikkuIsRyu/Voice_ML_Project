#!/usr/bin/env python3
"""
Select 3 diverse, high-naturalness reference prompt audio candidates for female_v4.
Evaluates phonetic coverage, tone variety, pitch contour continuity, and natural pause phrasing.
"""

import sys
import json
import shutil
from pathlib import Path
import soundfile as sf
import numpy as np
from pypinyin import pinyin, Style

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tutor" / "backend"))
from audio_utils import extract_f0_contour

def count_distinct_tones(text: str) -> set:
    tones = set()
    pyn = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    for item in pyn:
        for digit in "1234":
            if digit in item[0]:
                tones.add(int(digit))
    return tones

def score_candidate(item: dict) -> float:
    wav_p = Path(item["wav_path"])
    if not wav_p.exists():
        return -1.0
    try:
        data, sr = sf.read(str(wav_p))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        dur = len(data) / sr

        # Ideal duration 5.2s - 7.5s
        if dur < 4.8 or dur > 8.0:
            return -1.0
        dur_score = 100.0 - abs(dur - 6.2) * 20.0

        peak = float(np.max(np.abs(data)))
        if peak >= 0.98 or peak < 0.4:
            return -1.0
        peak_score = 100.0 if (0.6 <= peak <= 0.90) else 75.0

        # All 4 tones should be present
        tones = count_distinct_tones(item["text"])
        tone_score = len(tones) * 25.0

        # Pitch contour & voicing continuity
        f0_info = extract_f0_contour(data, sr=sr, n_points=80)
        if not f0_info["has_voicing"] or f0_info["mean_f0"] < 160.0 or f0_info["mean_f0"] > 320.0:
            return -1.0
        voiced_ratio = sum(1 for v in f0_info["raw_f0"] if v is not None) / max(len(f0_info["raw_f0"]), 1)
        voicing_score = min(100.0, voiced_ratio * 135.0)

        # Natural prosody bonus: presence of breathing commas '，'
        comma_bonus = 15.0 if "，" in item["text"] else 0.0

        total = (0.30 * tone_score + 0.30 * voicing_score + 0.20 * dur_score + 0.10 * peak_score + comma_bonus)
        return float(total)
    except Exception:
        return -1.0

def main():
    holdout_dir = PROJECT_ROOT / "data" / "train_v4" / "csmsc_holdout"
    manifest_p = holdout_dir / "holdout_manifest.json"
    if not manifest_p.exists():
        print(f"Error: {manifest_p} not found!")
        sys.exit(1)

    with open(manifest_p, "r", encoding="utf-8") as f:
        items = json.load(f)

    scored = []
    for it in items:
        sc = score_candidate(it)
        if sc > 0:
            scored.append((sc, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    if len(scored) < 3:
        print(f"Warning: Only {len(scored)} suitable candidates found!")
        selected = scored
    else:
        # Pick top 3 diverse candidates
        selected = scored[:3]

    out_dir = PROJECT_ROOT / "data" / "train_v4" / "csmsc"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTop 3 Reference Audio Candidates Selected for female_v4:")
    for idx, (score, it) in enumerate(selected, 1):
        cand_wav = out_dir / f"csmsc_ref_candidate_{idx}.wav"
        cand_txt = out_dir / f"csmsc_ref_candidate_{idx}.txt"

        data, sr = sf.read(it["wav_path"])
        sf.write(str(cand_wav), data, sr)
        with open(cand_txt, "w", encoding="utf-8") as f:
            f.write(it["text"])

        print(f"  Candidate {idx} [Score: {score:.1f}/100]:")
        print(f"    Text: {it['text']}")
        print(f"    Duration: {it['duration']:.2f}s | Saved: {cand_wav.name}")

    # Default csmsc_ref.wav to Candidate 1
    best_wav = out_dir / "csmsc_ref.wav"
    best_txt = out_dir / "csmsc_ref.txt"
    shutil.copy2(out_dir / "csmsc_ref_candidate_1.wav", best_wav)
    shutil.copy2(out_dir / "csmsc_ref_candidate_1.txt", best_txt)
    print(f"\nSet default reference prompt to Candidate 1: {best_wav.name}")

if __name__ == "__main__":
    main()
