#!/usr/bin/env python3
"""
Synthesize Dual-Audio Vocabulary (1.0x and 0.65x) with female_v4 for Voice-ML v4.

Features:
1. Iterates over vocabulary words from data/vocab/*.txt (or specified lessons).
2. Switches GPT-SoVITS weights to female_v4 (CSMSC 7.5h Studio Voice).
3. Synthesizes 1.0x normal audio: output_v2/vocab/<lesson>/...
4. Synthesizes 0.65x slow audio: output_v2_slow/vocab/<lesson>/...
5. Automatically saves both naming conventions (with and without prefix) for Anki and Tutor compatibility.
6. Pads short single/double characters with period for acoustic stability.
7. Skips valid cached files unless --force is specified.
"""

import argparse
import io
import json
import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import yaml
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
GPT_SOVITS_DIR = PROJECT_ROOT / "GPT-SoVITS"

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
    proc = subprocess.Popen(cmd, cwd=str(GPT_SOVITS_DIR), env=env)
    for _ in range(40):
        time.sleep(2)
        if is_api_alive():
            print("API server ready.")
            return proc
    raise RuntimeError("Timed out starting API server.")


def set_voice_weights(sovits_path: str, gpt_path: str):
    sovits_abs = str((GPT_SOVITS_DIR / sovits_path).resolve())
    gpt_abs = str((GPT_SOVITS_DIR / gpt_path).resolve())
    r1 = requests.get(f"{API_URL}/set_sovits_weights", params={"weights_path": sovits_abs}, timeout=20)
    r2 = requests.get(f"{API_URL}/set_gpt_weights", params={"weights_path": gpt_abs}, timeout=20)
    if r1.status_code != 200 or r2.status_code != 200:
        raise RuntimeError(f"Weight switch failed: sovits={r1.text}, gpt={r2.text}")
    print(f"Switched model weights -> {Path(sovits_path).stem} | {Path(gpt_path).stem}")


def synthesize_word_audio(text: str, out_path: Path, ref_wav: Path, ref_txt: str, speed: float = 1.0) -> bool:
    """Synthesize clean vocabulary audio via TTS API."""
    clean_chars = re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", text)
    synth_text = text
    # Pad very short words with full stop to avoid truncation in GPT-SoVITS
    if len(clean_chars) <= 2 and not any(p in text for p in "，。？！"):
        synth_text = text + "。"

    payload = {
        "text": synth_text,
        "text_lang": "zh",
        "ref_audio_path": str(ref_wav.resolve()),
        "prompt_text": ref_txt,
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
    parser = argparse.ArgumentParser(description="Synthesize Dual-Audio Vocab with female_v4")
    parser.add_argument("--voice", default="female_v4", help="Voice profile name from voice_profiles.yaml (default: female_v4)")
    parser.add_argument("--lessons", nargs="+", help="Specific lessons to synthesize (e.g. lesson07 lesson08)")
    parser.add_argument("--force", action="store_true", help="Force re-synthesis of existing files")
    parser.add_argument("--limit", type=int, default=0, help="Limit synthesis count (0 for all)")
    args = parser.parse_args()

    print("=" * 65)
    print(f"VOICE-ML v4: DUAL-AUDIO VOCABULARY SYNTHESIS ({args.voice})")
    print("=" * 65)

    # 1. Load voice profile
    with open(PROJECT_ROOT / "configs" / "voice_profiles.yaml", "r", encoding="utf-8") as f:
        voice_cfg = yaml.safe_load(f)["voices"][args.voice]

    ref_wav = PROJECT_ROOT / voice_cfg["ref_audio"]
    ref_txt = voice_cfg["prompt_text"]

    # 2. Ensure API server is alive and weights loaded
    api_proc = ensure_api_running()
    set_voice_weights(voice_cfg["sovits_weights"], voice_cfg["gpt_weights"])

    # 3. Collect entries
    txt_files = sorted(VOCAB_DIR.glob("*.txt"))
    if args.lessons:
        target_lessons = set(args.lessons)
        txt_files = [tf for tf in txt_files if tf.stem in target_lessons]

    all_entries = []
    for txt in txt_files:
        if txt.name.endswith("_items.txt") or txt.name == "all_vocab_dump.json":
            continue
        lesson_code, topic, entries = parse_lesson_file(txt)
        for e in entries:
            all_entries.append((lesson_code, e["line_idx"], e["hanzi"]))

    print(f"Total vocabulary words to synthesize: {len(all_entries)} across {len(txt_files)} lessons.")
    if args.limit > 0:
        all_entries = all_entries[:args.limit]

    count_norm_done = 0
    count_slow_done = 0
    start_time = time.time()

    for idx, (lesson_code, line_idx, hanzi) in enumerate(all_entries, start=1):
        # Normal (1.0x)
        norm_v3_path = OUT_NORM_BASE / lesson_code / f"v3_norm_{lesson_code}_{line_idx:03d}_{hanzi}.wav"
        norm_plain_path = OUT_NORM_BASE / lesson_code / f"{lesson_code}_{line_idx:03d}_{hanzi}.wav"
        
        # Slow (0.65x)
        slow_v3_path = OUT_SLOW_BASE / lesson_code / f"v3_slow_{lesson_code}_{line_idx:03d}_{hanzi}.wav"
        slow_plain_path = OUT_SLOW_BASE / lesson_code / f"{lesson_code}_{line_idx:03d}_{hanzi}.wav"

        # 1. Normal Audio
        need_norm = args.force or not (norm_v3_path.exists() and norm_plain_path.exists())
        if need_norm:
            ok = synthesize_word_audio(hanzi, norm_v3_path, ref_wav=ref_wav, ref_txt=ref_txt, speed=1.0)
            if ok:
                count_norm_done += 1
                try:
                    shutil.copy2(norm_v3_path, norm_plain_path)
                except Exception:
                    pass

        # 2. Slow Audio (0.65x)
        need_slow = args.force or not (slow_v3_path.exists() and slow_plain_path.exists())
        if need_slow:
            ok = synthesize_word_audio(hanzi, slow_v3_path, ref_wav=ref_wav, ref_txt=ref_txt, speed=0.65)
            if ok:
                count_slow_done += 1
                try:
                    shutil.copy2(slow_v3_path, slow_plain_path)
                except Exception:
                    pass

        if idx % 10 == 0 or idx == len(all_entries):
            print(f"  [{idx}/{len(all_entries)}] {lesson_code} - {hanzi} (new 1.0x: {count_norm_done}, new 0.65x: {count_slow_done})")

    elapsed = time.time() - start_time
    print(f"\n✓ Dual-Audio Vocab Synthesis complete in {elapsed:.1f}s.")
    print(f"  New 1.0x files synthesized: {count_norm_done}")
    print(f"  New 0.65x files synthesized: {count_slow_done}")


if __name__ == "__main__":
    main()
