#!/usr/bin/env python3
"""
Synthesize Dual-Audio Vocabulary (1.0x and 0.65x) with female_v3 for Voice-ML v3 (Phase 4).

Features:
1. Iterates over all 345 vocabulary words from data/vocab/*.txt.
2. Synthesizes 1.0x normal audio: output_v2/vocab/<lesson>/v3_norm_<lesson>_<idx:03d>_<hanzi>.wav.
3. Synthesizes 0.65x slow audio: output_v2_slow/vocab/<lesson>/v3_slow_<lesson>_<idx:03d>_<hanzi>.wav.
4. Uses female_v3 (CSMSC 2.5h) via GPT-SoVITS API.
5. Pads short single/double characters with period for acoustic stability.
6. Automatically skips valid cached files.
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import requests
import soundfile as sf

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_DIR = PROJECT_ROOT / "data" / "vocab"
OUT_NORM_BASE = PROJECT_ROOT / "output_v2" / "vocab"
OUT_SLOW_BASE = PROJECT_ROOT / "output_v2_slow" / "vocab"
API_URL = "http://127.0.0.1:9880"
DEFAULT_REF_AUDIO = PROJECT_ROOT / "data" / "train_v3" / "csmsc" / "csmsc_ref.wav"
DEFAULT_PROMPT_TEXT = "帛书，化为，袅袅青烟，飘向，九天重霄，告慰，始祖先灵。"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from export_anki_deck import parse_lesson_file


def is_api_alive() -> bool:
    try:
        r = requests.get(f"{API_URL}/docs", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def ensure_api_running():
    if is_api_alive():
        return None
    print("Starting GPT-SoVITS API server for vocab synthesis...")
    import subprocess
    env = os.environ.copy()
    conda_env = r"D:\miniconda3\envs\GPTSoVits"
    env["PATH"] = rf"{conda_env}\Lib\site-packages\torch\lib;{conda_env};{conda_env}\Scripts;" + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        sys.executable,
        "api_v2.py",
        "-a", "127.0.0.1",
        "-p", "9880",
        "-c", "GPT_SoVITS/configs/tts_infer.yaml"
    ]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT / "GPT-SoVITS"), env=env)
    for _ in range(40):
        time.sleep(2)
        if is_api_alive():
            print("API server ready.")
            return proc
    raise RuntimeError("Timed out starting API server.")


def synthesize_word_audio(text: str, out_path: Path, speed: float = 1.0) -> bool:
    """Synthesize clean vocabulary audio via TTS API."""
    clean_chars = re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", text)
    synth_text = text
    # Pad very short words with full stop to avoid truncation in GPT-SoVITS
    if len(clean_chars) <= 2 and not any(p in text for p in "，。？！"):
        synth_text = text + "。"

    payload = {
        "text": synth_text,
        "text_lang": "zh",
        "ref_audio_path": str(DEFAULT_REF_AUDIO.resolve()),
        "prompt_text": DEFAULT_PROMPT_TEXT,
        "prompt_lang": "zh",
        "speed": speed,
        "speed_factor": speed,
        "top_k": 10,
        "top_p": 1.0,
        "temperature": 0.8
    }
    try:
        r = requests.post(f"{API_URL}/tts", json=payload, timeout=120)
        if r.status_code == 200 and len(r.content) > 500:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True
        else:
            print(f"  Error {r.status_code} for '{text}': {r.text[:80]}")
            return False
    except Exception as e:
        print(f"  Exception during synthesis for '{text}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Synthesize Dual-Audio Vocab with female_v3")
    parser.add_argument("--force", action="store_true", help="Force re-synthesis of existing files")
    parser.add_argument("--limit", type=int, default=0, help="Limit synthesis count (0 for all)")
    args = parser.parse_args()

    print("=" * 65)
    print("PHASE 4: DUAL-AUDIO VOCABULARY SYNTHESIS (1.0x & 0.65x)")
    print("=" * 65)

    api_proc = ensure_api_running()

    txt_files = sorted(VOCAB_DIR.glob("*.txt"))
    all_entries = []
    for txt in txt_files:
        if txt.name.endswith("_items.txt") or txt.name == "all_vocab_dump.json":
            continue
        lesson_code, topic, entries = parse_lesson_file(txt)
        for e in entries:
            all_entries.append((lesson_code, e["line_idx"], e["hanzi"]))

    print(f"Total vocabulary words: {len(all_entries)}")
    if args.limit > 0:
        all_entries = all_entries[:args.limit]

    count_norm_done = 0
    count_slow_done = 0
    start_time = time.time()

    for idx, (lesson_code, line_idx, hanzi) in enumerate(all_entries, start=1):
        norm_path = OUT_NORM_BASE / lesson_code / f"v3_norm_{lesson_code}_{line_idx:03d}_{hanzi}.wav"
        slow_path = OUT_SLOW_BASE / lesson_code / f"v3_slow_{lesson_code}_{line_idx:03d}_{hanzi}.wav"

        # 1. Normal (1.0x)
        need_norm = args.force or not norm_path.exists()
        if need_norm:
            ok = synthesize_word_audio(hanzi, norm_path, speed=1.0)
            if ok:
                count_norm_done += 1
        
        # 2. Slow (0.65x)
        need_slow = args.force or not slow_path.exists()
        if need_slow:
            ok = synthesize_word_audio(hanzi, slow_path, speed=0.65)
            if ok:
                count_slow_done += 1

        if idx % 25 == 0 or idx == len(all_entries):
            print(f"  [{idx}/{len(all_entries)}] Processed: {hanzi} (new norm: {count_norm_done}, new slow: {count_slow_done})")

    elapsed = time.time() - start_time
    print(f"\n✓ Dual-Audio Vocab Synthesis complete in {elapsed:.1f}s.")
    print(f"  New 1.0x files synthesized: {count_norm_done}")
    print(f"  New 0.65x files synthesized: {count_slow_done}")


if __name__ == "__main__":
    main()
