#!/usr/bin/env python3
import os
import sys
import re
import json
from pathlib import Path
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BZNSYP_DIR = PROJECT_ROOT.parent / "Voice_models" / "datasets" / "BZNSYP"
V4_CSMSC_DIR = PROJECT_ROOT / "data" / "train_v4" / "csmsc"
HOLDOUT_DIR = PROJECT_ROOT / "data" / "train_v4" / "csmsc_holdout"

def clean_prosody_text(raw_text: str) -> str:
    t = raw_text.strip()
    t = t.replace("#3", "，").replace("#4", "。").replace("#2", "，").replace("#1", "")
    t = re.sub(r"[，,]+", "，", t)
    t = re.sub(r"[。.]+", "。", t)
    t = re.sub(r"[，,][。.]", "。", t)
    t = re.sub(r"^[，。？！、\s]+", "", t)
    t = t.strip()
    if t and not any(t.endswith(p) for p in "。？！…"):
        t += "。"
    return t

def main():
    label_files = list(BZNSYP_DIR.glob("**/000001-010000.txt"))
    label_path = label_files[0]
    raw_items = {}
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 2):
        parts = lines[i].split("\t")
        if len(parts) >= 2:
            wid = parts[0].strip()
            raw_items[wid] = {
                "id": wid,
                "text": clean_prosody_text(parts[1].strip()),
                "pinyin": lines[i+1].strip() if i+1 < len(lines) else ""
            }

    # 1. Holdout
    holdout_wavs = sorted((HOLDOUT_DIR / "wavs").glob("*.wav"))
    holdout_items = []
    for hw in holdout_wavs:
        wid = hw.stem
        if wid in raw_items:
            info = sf.info(str(hw))
            holdout_items.append({
                "id": wid,
                "wav_path": str(hw).replace("\\", "/"),
                "text": raw_items[wid]["text"],
                "duration": info.duration
            })

    with open(HOLDOUT_DIR / "holdout_manifest.json", "w", encoding="utf-8") as f:
        json.dump(holdout_items, f, ensure_ascii=False, indent=2)
    print(f"Wrote holdout manifest with {len(holdout_items)} items.")

    # 2. Train set: exactly 7.50 hours (27,000s)
    train_wavs = sorted((V4_CSMSC_DIR / "wavs").glob("*.wav"))
    print(f"Total wavs on disk in train_v4/csmsc/wavs: {len(train_wavs)}")

    target_s = 7.50 * 3600.0  # 27,000s
    accumulated_s = 0.0
    selected = []

    for tw in train_wavs:
        wid = tw.stem
        if wid not in raw_items:
            continue
        info = sf.info(str(tw))
        if info.duration < 1.8:
            continue
        selected.append({
            "id": wid,
            "wav_path": str(tw).replace("\\", "/"),
            "text": raw_items[wid]["text"],
            "duration": info.duration
        })
        accumulated_s += info.duration
        if accumulated_s >= target_s:
            break

    print(f"Selected {len(selected)} clips totaling {accumulated_s/3600:.2f} hours ({accumulated_s:.1f} s).")

    manifest_p = V4_CSMSC_DIR / "2-name2text.txt"
    with open(manifest_p, "w", encoding="utf-8") as f:
        for it in selected:
            f.write(f"{it['wav_path']}|female_v4|zh|{it['text']}\n")

    print(f"Successfully wrote training manifest: {manifest_p}")

if __name__ == "__main__":
    main()
