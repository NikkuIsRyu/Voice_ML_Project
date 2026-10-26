#!/usr/bin/env python3
"""
Fast parallel 7.50-hour CSMSC dataset builder for female_v4.
Utilizes 8 worker processes to trim, normalize, and format audio clips.
"""

import os
import sys
import re
import json
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import soundfile as sf
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tutor" / "backend"))
from audio_utils import trim_speech

BZNSYP_DIR = (PROJECT_ROOT.parent / "Voice_models" / "datasets" / "BZNSYP").resolve()
V3_CSMSC_DIR = PROJECT_ROOT / "data" / "train_v3" / "csmsc"
V4_CSMSC_DIR = PROJECT_ROOT / "data" / "train_v4" / "csmsc"
HOLDOUT_DIR = PROJECT_ROOT / "data" / "train_v4" / "csmsc_holdout"

TARGET_TOTAL_SECONDS = 7.50 * 3600.0  # 27,000.0 seconds

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

def normalize_loudness(audio_data: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms < 1e-6:
        return audio_data
    target_rms = 10.0 ** (target_lufs / 20.0)
    gain = target_rms / rms
    norm = audio_data * gain
    peak = np.max(np.abs(norm))
    if peak > 0.891:
        norm = norm * (0.891 / peak)
    return norm

def process_single_clip(task_tuple):
    wid, item, src_wav_str, tgt_wav_str, is_holdout = task_tuple
    src_wav = Path(src_wav_str)
    tgt_wav = Path(tgt_wav_str)

    if not src_wav.exists():
        return None

    try:
        # If target already exists and is non-empty, avoid reprocessing
        if tgt_wav.exists() and tgt_wav.stat().st_size > 1000:
            info = sf.info(str(tgt_wav))
            return {
                "id": wid,
                "wav_path": str(tgt_wav).replace("\\", "/"),
                "text": item["text"],
                "duration": info.duration,
                "is_holdout": is_holdout
            }

        data, sr = sf.read(str(src_wav))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = data.astype(np.float32)

        if np.max(np.abs(data)) >= 0.999:
            return None

        trimmed_data, vad_meta = trim_speech(data, sr=sr, top_db=30.0)
        dur = len(trimmed_data) / sr

        if dur < 2.0 or dur > 9.5 or vad_meta["speech_ratio"] < 0.50:
            return None

        norm_data = normalize_loudness(trimmed_data, sr, target_lufs=-23.0)
        tgt_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(tgt_wav), norm_data, sr)

        return {
            "id": wid,
            "wav_path": str(tgt_wav).replace("\\", "/"),
            "text": item["text"],
            "duration": dur,
            "is_holdout": is_holdout
        }
    except Exception:
        return None

def main():
    wav_out = V4_CSMSC_DIR / "wavs"
    wav_out.mkdir(parents=True, exist_ok=True)
    holdout_wav_out = HOLDOUT_DIR / "wavs"
    holdout_wav_out.mkdir(parents=True, exist_ok=True)

    label_files = list(BZNSYP_DIR.glob("**/000001-010000.txt"))
    label_path = label_files[0]
    wave_dir = list(BZNSYP_DIR.glob("**/Wave"))[0]

    raw_items = {}
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 2):
        parts = lines[i].split("\t")
        if len(parts) >= 2:
            wid = parts[0].strip()
            raw_items[wid] = {
                "id": wid,
                "raw_text": parts[1].strip(),
                "text": clean_prosody_text(parts[1].strip()),
                "pinyin": lines[i+1].strip() if i+1 < len(lines) else ""
            }

    # 1. Reuse existing v3 clips
    v3_manifest = V3_CSMSC_DIR / "2-name2text.txt"
    v3_items = []
    v3_ids = set()
    total_duration = 0.0

    print("Checking migrated v3 clips...")
    with open(v3_manifest, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            src_p = Path(parts[0])
            wid = src_p.stem
            v3_ids.add(wid)
            dest_wav = wav_out / f"{wid}.wav"
            if not dest_wav.exists() or dest_wav.stat().st_size < 1000:
                shutil.copy2(src_p, dest_wav)
            info = sf.info(str(dest_wav))
            dur = info.duration
            total_duration += dur
            v3_items.append({
                "id": wid,
                "wav_path": str(dest_wav).replace("\\", "/"),
                "text": parts[3],
                "duration": dur,
                "is_holdout": False
            })

    print(f"Migrated {len(v3_items)} v3 clips ({total_duration/3600:.2f}h).")
    needed_seconds = TARGET_TOTAL_SECONDS - total_duration
    print(f"Need ~{needed_seconds/3600:.2f}h more from remaining CSMSC...")

    remaining_ids = sorted([wid for wid in raw_items.keys() if wid not in v3_ids])
    tasks = []
    holdout_counter = 0

    for idx, wid in enumerate(remaining_ids):
        # Reserve holdout clips
        is_holdout = (holdout_counter < 100 and idx % 25 == 0)
        if is_holdout:
            holdout_counter += 1
            tgt = str(holdout_wav_out / f"{wid}.wav")
        else:
            tgt = str(wav_out / f"{wid}.wav")
        tasks.append((wid, raw_items[wid], str(wave_dir / f"{wid}.wav"), tgt, is_holdout))

    print(f"Launching parallel processing across 8 CPU worker processes ({len(tasks)} tasks)...")
    all_selected = list(v3_items)
    holdout_items = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_clip, t) for t in tasks]
        for f in as_completed(futures):
            res = f.result()
            if res is None:
                continue
            if res["is_holdout"]:
                holdout_items.append(res)
            else:
                if total_duration < TARGET_TOTAL_SECONDS:
                    all_selected.append(res)
                    total_duration += res["duration"]
                    if len(all_selected) % 500 == 0:
                        print(f"[{len(all_selected)} clips] Total duration: {total_duration/60:.1f}m ({total_duration/3600:.2f}h) | Holdout: {len(holdout_items)}")

    print(f"\nCompleted v4 CSMSC Dataset Creation:")
    print(f"  Total training clips : {len(all_selected)}")
    print(f"  Total duration       : {total_duration/3600:.2f} hours ({total_duration:.1f} s)")
    print(f"  Holdout clips        : {len(holdout_items)}")

    # Sort all_selected by ID for clean manifest
    all_selected.sort(key=lambda x: x["id"])

    manifest_path = V4_CSMSC_DIR / "2-name2text.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for it in all_selected:
            f.write(f"{it['wav_path']}|female_v4|zh|{it['text']}\n")
    print(f"Wrote training manifest: {manifest_path}")

    holdout_items.sort(key=lambda x: x["id"])
    holdout_manifest = HOLDOUT_DIR / "holdout_manifest.json"
    with open(holdout_manifest, "w", encoding="utf-8") as f:
        json.dump(holdout_items, f, ensure_ascii=False, indent=2)
    print(f"Wrote holdout manifest: {holdout_manifest}")

if __name__ == "__main__":
    main()
