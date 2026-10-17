#!/usr/bin/env python3
"""
Generate contextual HSK1/2 example sentences and audio for Voice-ML v3 (Improvement 2.4).

Features:
1. Loads all 345 vocabulary words from data/vocab/*.txt.
2. Formats natural example sentences with <b>...</b> highlighting, tone-marked pinyin, and German translations.
3. Saves versioned JSON manifest: data/vocab/vocab_example_sentences.json.
4. Synthesizes high-fidelity audio via female_v3 (CSMSC 2.5h) to output_v2/examples/ex_<lesson>_<idx:03d>_<hanzi>.wav.
5. Verifies audio validity and length.
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests
import soundfile as sf
from pypinyin import pinyin, Style

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_DIR = PROJECT_ROOT / "data" / "vocab"
OUTPUT_EXAMPLES_DIR = PROJECT_ROOT / "output_v2" / "examples"
API_URL = "http://127.0.0.1:9880"
DEFAULT_REF_AUDIO = PROJECT_ROOT / "data" / "train_v3" / "csmsc" / "csmsc_ref.wav"
DEFAULT_PROMPT_TEXT = "帛书，化为，袅袅青烟，飘向，九天重霄，告慰，始祖先灵。"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from vocab_sentences_catalog import CATALOG


def format_sentence_pinyin(raw_sentence: str) -> str:
    """Generate tone-marked Pinyin for a Chinese sentence preserving punctuation."""
    clean = re.sub(r"<[^>]+>", "", raw_sentence)
    tokens = []
    current_hanzi = []

    for char in clean:
        if '\u4e00' <= char <= '\u9fff':
            current_hanzi.append(char)
        else:
            if current_hanzi:
                py_list = [p[0] for p in pinyin("".join(current_hanzi), style=Style.TONE)]
                tokens.append(" ".join(py_list))
                current_hanzi = []
            tokens.append(char)

    if current_hanzi:
        py_list = [p[0] for p in pinyin("".join(current_hanzi), style=Style.TONE)]
        tokens.append(" ".join(py_list))

    res = "".join(tokens)
    res = re.sub(r"\s+([，。？！、；：,!?.])", r"\1", res)
    res = re.sub(r"([，。？！、；：,!?])\s*", r"\1 ", res).strip()
    return res


def load_all_vocab_items() -> List[Dict[str, Any]]:
    """Load all 345 vocab items deterministically."""
    from export_anki_deck import parse_lesson_file
    items = []
    for txt_file in sorted(VOCAB_DIR.glob("*.txt")):
        if txt_file.name.endswith("_items.txt") or txt_file.name == "all_vocab_dump.json":
            continue
        code, topic, entries = parse_lesson_file(txt_file)
        for e in entries:
            items.append({
                "lesson": code,
                "topic": topic,
                "line_idx": e["line_idx"],
                "hanzi": e["hanzi"],
                "raw_pinyin": e["raw_pinyin"],
                "german": e["german"]
            })
    return items


def is_api_alive() -> bool:
    try:
        r = requests.get(f"{API_URL}/docs", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def ensure_api_running():
    if is_api_alive():
        return None
    print("Starting GPT-SoVITS API server for example synthesis...")
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


def synthesize_example_audio(text: str, out_path: Path, speed: float = 1.0) -> bool:
    """Synthesize clean example sentence via running TTS API."""
    clean_text = re.sub(r"<[^>]+>", "", text).strip()
    if not clean_text:
        return False

    payload = {
        "text": clean_text,
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
            print(f"  Error {r.status_code}: {r.text[:80]}")
            return False
    except Exception as e:
        print(f"  Exception during synthesis: {e}")
        return False


def build_manifest(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the structured list of all 345 example sentences."""
    manifest = []
    for it in items:
        lesson = it["lesson"]
        line_idx = it["line_idx"]
        hanzi = it["hanzi"]
        raw_pinyin = it["raw_pinyin"]
        german = it["german"]

        key = (lesson, line_idx, hanzi)
        if key in CATALOG:
            sentence_hanzi, sentence_de = CATALOG[key]
        else:
            raise KeyError(f"Missing catalog entry for: {key}")

        clean_sentence = re.sub(r"<[^>]+>", "", sentence_hanzi)
        sentence_pinyin = format_sentence_pinyin(sentence_hanzi)
        audio_filename = f"ex_{lesson}_{line_idx:03d}_{hanzi}.wav"

        manifest.append({
            "lesson": lesson,
            "line_idx": line_idx,
            "hanzi": hanzi,
            "raw_pinyin": raw_pinyin,
            "german": german,
            "sentence_hanzi": sentence_hanzi,
            "clean_sentence": clean_sentence,
            "sentence_pinyin": sentence_pinyin,
            "sentence_translation_de": sentence_de,
            "audio_filename": audio_filename
        })
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate contextual example sentences and audio (Phase 5)")
    parser.add_argument("--synthesize", action="store_true", default=True, help="Synthesize audio files (default: True)")
    parser.add_argument("--no-synthesize", dest="synthesize", action="store_false", help="Skip audio synthesis")
    parser.add_argument("--force", action="store_true", help="Overwrite existing audio files")
    parser.add_argument("--limit", type=int, default=0, help="Limit synthesis count (0 for all)")
    args = parser.parse_args()

    print("=" * 65)
    print("PHASE 5: AUTOMATIC EXAMPLE SENTENCES GENERATION (2.4)")
    print("=" * 65)

    items = load_all_vocab_items()
    print(f"Loaded {len(items)} vocabulary words from {VOCAB_DIR}")
    assert len(items) == 345, f"Expected 345 words, got {len(items)}"

    manifest = build_manifest(items)
    print(f"Built manifest for {len(manifest)} example sentences.")

    # Save to data/vocab/vocab_example_sentences.json
    manifest_path = VOCAB_DIR / "vocab_example_sentences.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved manifest to: {manifest_path} ({manifest_path.stat().st_size} bytes)")

    if args.synthesize:
        print("\nStarting TTS synthesis with female_v3...")
        api_proc = ensure_api_running()

        OUTPUT_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        count_synth = 0
        count_skipped = 0
        total_to_process = manifest if args.limit <= 0 else manifest[:args.limit]

        start_time = time.time()
        for idx, entry in enumerate(total_to_process, start=1):
            out_file = OUTPUT_EXAMPLES_DIR / entry["audio_filename"]
            if out_file.exists() and not args.force:
                try:
                    data, sr = sf.read(str(out_file))
                    if len(data) > 1000:
                        count_skipped += 1
                        if idx % 50 == 0 or idx == len(total_to_process):
                            print(f"  [{idx}/{len(total_to_process)}] Cached: {out_file.name}")
                        continue
                except Exception:
                    pass

            print(f"  [{idx}/{len(total_to_process)}] Synthesizing: {entry['clean_sentence']} -> {out_file.name}")
            ok = synthesize_example_audio(entry["clean_sentence"], out_file, speed=1.0)
            if not ok:
                print(f"    WARNING: Failed synthesis for {out_file.name}")
            else:
                count_synth += 1

        elapsed = time.time() - start_time
        print(f"\nSynthesis finished in {elapsed:.1f}s.")
        print(f"  Synthesized: {count_synth}")
        print(f"  Cached/Skipped: {count_skipped}")
        print(f"  Total valid files in {OUTPUT_EXAMPLES_DIR}: {len(list(OUTPUT_EXAMPLES_DIR.glob('*.wav')))}")

    print("\n✓ Phase 5 manifest generation and synthesis complete.")


if __name__ == "__main__":
    main()
