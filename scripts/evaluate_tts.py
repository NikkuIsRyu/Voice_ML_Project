#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_tts.py — Comprehensive Evaluation Harness for GPT-SoVITS TTS Models.

Measures:
- Synthesis success rate
- Audio duration & character rate (clips too short or hallucinated)
- Character Error Rate (CER) via Faster-Whisper ASR
- Pinyin & Tone match percentage via pypinyin
- Category-level breakdown (Vocab, Dialogue, Complex)
- Generates markdown report & interactive HTML player
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows & add torch CUDA dlls to PATH for ctranslate2
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

import requests
import soundfile as sf
import pypinyin
from faster_whisper import WhisperModel


def strip_punct(text: str) -> str:
    """Remove all punctuation and whitespace for pure Chinese character comparison."""
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", text)


def get_pinyin_tones(text: str):
    """Return list of pinyin syllables with tone numbers (e.g. ['ni3', 'hao3'])."""
    clean = strip_punct(text)
    if not clean:
        return []
    pinyin_list = pypinyin.pinyin(clean, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)
    return [item[0] for item in pinyin_list]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate character-level edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


def compute_cer(target: str, hyp: str) -> float:
    t_clean = strip_punct(target)
    h_clean = strip_punct(hyp)
    if not t_clean:
        return 0.0 if not h_clean else 1.0
    dist = levenshtein_distance(t_clean, h_clean)
    return min(dist / len(t_clean), 1.0)


def compute_tone_match(target: str, hyp: str) -> float:
    t_pinyins = get_pinyin_tones(target)
    h_pinyins = get_pinyin_tones(hyp)
    if not t_pinyins:
        return 1.0
    matches = sum(1 for t, h in zip(t_pinyins, h_pinyins) if t == h)
    return matches / max(len(t_pinyins), len(h_pinyins))


def parse_testset(filepath: str):
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                item_id, category, text = parts[0], parts[1], parts[2]
                notes = parts[3] if len(parts) > 3 else ""
                items.append({
                    "id": item_id,
                    "category": category,
                    "text": text,
                    "notes": notes
                })
    return items


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-SoVITS TTS output")
    parser.add_argument("--testset", default="eval/testset.txt", help="Path to testset.txt")
    parser.add_argument("--api_url", default="http://127.0.0.1:9880", help="TTS API Base URL")
    parser.add_argument("--ref_audio", default="data/reference_audio/A4_137.wav", help="Reference audio path")
    parser.add_argument("--ref_text", default="不弃雅不避俗化俗为雅化雅为俗俗不伤雅雅俗共赏。", help="Reference audio text")
    parser.add_argument("--ref_lang", default="zh", help="Reference language")
    parser.add_argument("--target_lang", default="zh", help="Target text language")
    parser.add_argument("--output_dir", default="eval/v1_baseline", help="Directory for eval results")
    parser.add_argument("--asr_dir", default="D:/.D Dokuments/Uni Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_models/asr/faster-whisper-large-v3", help="Faster-Whisper model path")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--skip_synth", action="store_true", help="Skip synthesis if audio already rendered")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    items = parse_testset(args.testset)
    print(f"Loaded {len(items)} items from {args.testset}")

    # 1. Synthesize Audio
    tts_endpoint = f"{args.api_url}/tts"
    ref_audio_abs = str(Path(args.ref_audio).resolve())

    if not args.skip_synth:
        print(f"Synthesizing items via TTS API at {tts_endpoint}...")
        for i, item in enumerate(items, 1):
            out_wav = audio_dir / f"{item['id']}.wav"
            item["wav_path"] = str(out_wav)
            if out_wav.exists() and out_wav.stat().st_size > 1000:
                continue

            # Payload matching api_v2.py specification
            synth_text = item["text"]
            if len(synth_text) <= 2 and not synth_text.endswith(("。", "！", "？", "，")):
                synth_text = synth_text + "。"

            payload = {
                "text": synth_text,
                "text_lang": args.target_lang,
                "ref_audio_path": ref_audio_abs,
                "prompt_text": args.ref_text,
                "prompt_lang": args.ref_lang,
                "top_k": args.top_k,
                "top_p": 1,
                "temperature": args.temperature,
                "text_split_method": "cut5" if len(synth_text) > 15 else "cut0",
                "batch_size": 1,
                "speed_factor": 1.0,
                "fragment_interval": 0.3,
                "split_bucket": True,
                "parallel_infer": True
            }

            try:
                t0 = time.time()
                resp = requests.post(tts_endpoint, json=payload, timeout=60)
                elapsed = time.time() - t0
                if resp.status_code == 200:
                    with open(out_wav, "wb") as f_out:
                        f_out.write(resp.content)
                    print(f"[{i}/{len(items)}] Synthesized {item['id']} ({elapsed:.2f}s): {item['text']}")
                else:
                    print(f"[{i}/{len(items)}] FAILED {item['id']}: HTTP {resp.status_code} - {resp.text[:100]}")
            except Exception as e:
                print(f"[{i}/{len(items)}] ERROR {item['id']}: {e}")
    else:
        for item in items:
            matches = list(audio_dir.glob(f"{item['id']}*.wav"))
            item["wav_path"] = str(matches[0]) if matches else ""

    # 2. Load ASR Model
    print(f"\nLoading Faster-Whisper ASR model from {args.asr_dir}...")
    try:
        asr_model = WhisperModel(args.asr_dir, device="cuda", compute_type="float16")
    except Exception as e:
        print(f"CUDA ASR initialization warning: {e}, falling back to int8 / CPU...")
        asr_model = WhisperModel(args.asr_dir, device="cpu", compute_type="int8")

    # 3. Evaluate audio files
    print("\nRunning evaluation on synthesized audio...")
    results = []
    for item in items:
        wav_path = item.get("wav_path")
        if not wav_path or not os.path.exists(wav_path):
            results.append({
                **item,
                "transcription": "",
                "cer": 1.0,
                "tone_match": 0.0,
                "duration": 0.0,
                "char_sec": 0.0,
                "flag": "MISSING_AUDIO"
            })
            continue

        try:
            audio_info = sf.info(wav_path)
            duration = audio_info.duration
        except Exception:
            duration = 0.0

        num_chars = max(len(strip_punct(item["text"])), 1)
        char_rate = duration / num_chars

        # Run Whisper transcription
        segments, _ = asr_model.transcribe(wav_path, language="zh", beam_size=5)
        transcript = "".join(seg.text for seg in segments).strip()

        cer = compute_cer(item["text"], transcript)
        tone_match = compute_tone_match(item["text"], transcript)

        # Sanity flags
        flag = "OK"
        if char_rate < 0.20:
            flag = "TOO_FAST_OR_CLIPPED"
        elif char_rate > 1.20 and len(item["text"]) < 5:
            flag = "DRAGGED_OR_HALLUCINATED"
        elif cer > 0.30:
            flag = "HIGH_CER"

        results.append({
            **item,
            "transcription": transcript,
            "cer": cer,
            "tone_match": tone_match,
            "duration": duration,
            "char_sec": char_rate,
            "flag": flag
        })

    # 4. Save CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,category,target_text,transcription,cer,tone_match,duration,char_sec,flag,notes\n")
        for r in results:
            clean_trans = r["transcription"].replace(",", "，")
            clean_target = r["text"].replace(",", "，")
            f.write(f"{r['id']},{r['category']},{clean_target},{clean_trans},{r['cer']:.4f},{r['tone_match']:.4f},{r['duration']:.2f},{r['char_sec']:.2f},{r['flag']},{r['notes']}\n")

    # 5. Compute Statistics & Markdown Summary
    valid_res = [r for r in results if r["flag"] != "MISSING_AUDIO"]
    total_count = len(valid_res)
    avg_cer = sum(r["cer"] for r in valid_res) / total_count if total_count else 0.0
    avg_tone = sum(r["tone_match"] for r in valid_res) / total_count if total_count else 0.0
    perfect_cer = sum(1 for r in valid_res if r["cer"] == 0.0) / total_count if total_count else 0.0

    cats = set(r["category"] for r in valid_res)
    cat_stats = {}
    for c in sorted(cats):
        sub = [r for r in valid_res if r["category"] == c]
        cat_stats[c] = {
            "count": len(sub),
            "cer": sum(r["cer"] for r in sub) / len(sub),
            "tone": sum(r["tone_match"] for r in sub) / len(sub),
            "perfect": sum(1 for r in sub if r["cer"] == 0.0) / len(sub)
        }

    # Top 10 worst items
    worst_items = sorted(valid_res, key=lambda x: (x["cer"], 1.0 - x["tone_match"]), reverse=True)[:10]

    summary_md = f"""# TTS Evaluation Report — {out_dir.name}

- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Total Test Items:** {len(items)} (Evaluated: {total_count})
- **Overall Mean CER:** **{avg_cer * 100:.2f}%**
- **Overall Tone Match Rate:** **{avg_tone * 100:.2f}%**
- **Exact Accuracy (CER = 0):** **{perfect_cer * 100:.2f}%**

---

### Category Performance Breakdown

| Category | Items | Mean CER | Tone Match | Exact Match Rate |
| :--- | :---: | :---: | :---: | :---: |
"""
    for c, s in cat_stats.items():
        summary_md += f"| **{c.capitalize()}** | {s['count']} | {s['cer']*100:.2f}% | {s['tone']*100:.2f}% | {s['perfect']*100:.2f}% |\n"

    summary_md += """
---

### Top 10 Problematic / Challenging Items

| ID | Category | Target Text | ASR Transcription | CER | Tone Match | Flag |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
"""
    for w in worst_items:
        summary_md += f"| {w['id']} | {w['category']} | {w['text']} | {w['transcription']} | {w['cer']*100:.1f}% | {w['tone_match']*100:.1f}% | `{w['flag']}` |\n"

    summary_md += """
---

### Duration & Pronunciation Notes
- Vocab items with 1-2 characters frequently need punctuation padding (e.g. `。`) or best-of-N selection.
- Polyphonic characters and sandhi shifts require accurate G2P normalization.
"""

    summary_path = out_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"\nEvaluation Complete!")
    print(f"- Overall CER: {avg_cer * 100:.2f}%")
    print(f"- Overall Tone Match: {avg_tone * 100:.2f}%")
    print(f"- Summary saved to: {summary_path}")
    print(f"- CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
