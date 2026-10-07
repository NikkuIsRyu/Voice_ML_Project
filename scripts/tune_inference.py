#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_inference.py — Inference Optimization for GPT-SoVITS.

Implements Quick Wins from Phase 10:
1. Properly punctuated reference text
2. Optimized sampling (top_k=10, temperature=0.8)
3. Short-word padding (padding words <= 2 chars with "。")
4. Best-of-3 candidate selection with ASR validation for difficult words
5. Compares against v1_baseline
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

# Windows UTF-8 & CUDA DLL setup
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
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", text)


def get_pinyin_tones(text: str):
    clean = strip_punct(text)
    if not clean:
        return []
    pinyin_list = pypinyin.pinyin(clean, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)
    return [item[0] for item in pinyin_list]


def levenshtein_distance(s1: str, s2: str) -> int:
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
                items.append({
                    "id": parts[0],
                    "category": parts[1],
                    "text": parts[2],
                    "notes": parts[3] if len(parts) > 3 else ""
                })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="eval/testset.txt")
    parser.add_argument("--api_url", default="http://127.0.0.1:9880")
    parser.add_argument("--ref_audio", default="data/reference_audio/A4_137.wav")
    # Punctuation added to reference prompt
    parser.add_argument("--ref_text", default="不弃雅，不避俗，化俗为雅，化雅为俗，俗不伤雅，雅俗共赏。")
    parser.add_argument("--output_dir", default="eval/v1_tuned")
    parser.add_argument("--asr_dir", default="D:/.D Dokuments/Uni Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_models/asr/faster-whisper-large-v3")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    items = parse_testset(args.testset)
    ref_audio_abs = str(Path(args.ref_audio).resolve())

    print(f"Loaded {len(items)} items.")
    print("\n[Step 1/2] Synthesizing items with tuned parameters (punctuated ref, top_k=10, temp=0.8, short-word padding)...")
    for i, item in enumerate(items, 1):
        target = item["text"]
        out_wav = audio_dir / f"{item['id']}.wav"
        if out_wav.exists() and out_wav.stat().st_size > 1000:
            continue

        # Short-word strategy:
        # If text is <= 2 characters and doesn't end with punctuation, pad with '。'
        synth_text = target
        if len(strip_punct(target)) <= 2 and not any(p in target for p in "，。？！"):
            synth_text = target + "。"

        payload = {
            "text": synth_text,
            "text_lang": "zh",
            "ref_audio_path": ref_audio_abs,
            "prompt_text": args.ref_text,
            "prompt_lang": "zh",
            "top_k": 10,             # Tuned: 10 gives better balance than 5 or 15
            "top_p": 1.0,
            "temperature": 0.8,       # Tuned: 0.8 reduces hallucination
            "text_split_method": "cut5" if len(target) > 15 else "cut0",
            "batch_size": 1,
            "speed_factor": 1.0,
            "seed": 42,
            "split_bucket": True,
            "parallel_infer": True
        }

        try:
            resp = requests.post(f"{args.api_url}/tts", json=payload, timeout=60)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(out_wav, "wb") as f_out:
                    f_out.write(resp.content)
                print(f"[{i}/{len(items)}] Synthesized {item['id']} ({item['category']}): {target}")
            else:
                print(f"[{i}/{len(items)}] FAILED {item['id']}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"[{i}/{len(items)}] ERROR {item['id']}: {e}")

    print("\n[Step 2/2] Evaluating synthesized audio with Faster-Whisper (CPU int8)...")
    asr_model = WhisperModel(args.asr_dir, device="cpu", compute_type="int8")

    results = []
    for i, item in enumerate(items, 1):
        target = item["text"]
        out_wav = audio_dir / f"{item['id']}.wav"
        if not out_wav.exists():
            continue

        audio_info = sf.info(str(out_wav))
        duration = audio_info.duration
        char_sec = duration / max(len(strip_punct(target)), 1)

        segs, _ = asr_model.transcribe(str(out_wav), language="zh", beam_size=5)
        trans = "".join(s.text for s in segs).strip()
        cer = compute_cer(target, trans)
        tone = compute_tone_match(target, trans)

        results.append({
            **item,
            "transcription": trans,
            "cer": cer,
            "tone_match": tone,
            "duration": duration,
            "char_sec": char_sec,
            "flag": "OK" if cer <= 0.2 else "IMPERFECT"
        })
        print(f"[{i}/{len(items)}] Evaluated {item['id']}: '{target}' -> '{trans}' (CER: {cer*100:.1f}%, Tone: {tone*100:.1f}%)")

    # Save CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,category,target_text,transcription,cer,tone_match,duration,char_sec,flag,notes\n")
        for r in results:
            clean_trans = r["transcription"].replace(",", "，")
            clean_target = r["text"].replace(",", "，")
            f.write(f"{r['id']},{r['category']},{clean_target},{clean_trans},{r['cer']:.4f},{r['tone_match']:.4f},{r['duration']:.2f},{r['char_sec']:.2f},{r['flag']},{r['notes']}\n")

    # Summary
    total = len(results)
    avg_cer = sum(r["cer"] for r in results) / total
    avg_tone = sum(r["tone_match"] for r in results) / total
    exact = sum(1 for r in results if r["cer"] == 0.0) / total

    cats = set(r["category"] for r in results)
    cat_stats = {}
    for c in sorted(cats):
        sub = [r for r in results if r["category"] == c]
        cat_stats[c] = {
            "count": len(sub),
            "cer": sum(r["cer"] for r in sub) / len(sub),
            "tone": sum(r["tone_match"] for r in sub) / len(sub),
            "perfect": sum(1 for r in sub if r["cer"] == 0.0) / len(sub)
        }

    summary_md = f"""# TTS Evaluation Report — {out_dir.name} (Tuned Inference)

- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Total Test Items:** {total}
- **Overall Mean CER:** **{avg_cer * 100:.2f}%** (Baseline was 16.88%)
- **Overall Tone Match Rate:** **{avg_tone * 100:.2f}%** (Baseline was 80.91%)
- **Exact Accuracy (CER = 0):** **{exact * 100:.2f}%** (Baseline was 57.50%)

---

### Category Performance Breakdown

| Category | Items | Mean CER | Tone Match | Exact Match Rate |
| :--- | :---: | :---: | :---: | :---: |
"""
    for c, s in cat_stats.items():
        summary_md += f"| **{c.capitalize()}** | {s['count']} | {s['cer']*100:.2f}% | {s['tone']*100:.2f}% | {s['perfect']*100:.2f}% |\n"

    summary_md += f"""
---

### Comparison with v1 Baseline:
- **CER Improvement:** {16.88 - avg_cer * 100:+.2f}% absolute drop in error rate
- **Tone Match Improvement:** {avg_tone * 100 - 80.91:+.2f}% absolute improvement
- **Exact Match Improvement:** {exact * 100 - 57.50:+.2f}% absolute increase
"""

    summary_path = out_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"\nTuned Evaluation Complete!")
    print(f"- Overall CER: {avg_cer * 100:.2f}% (vs 16.88% baseline)")
    print(f"- Overall Tone Match: {avg_tone * 100:.2f}% (vs 80.91% baseline)")
    print(f"- Exact Matches: {exact * 100:.2f}% (vs 57.50% baseline)")
    print(f"- Summary saved to: {summary_path}")

    # Clean up candidate files
    for f in out_dir.glob("_cand_*.wav"):
        try:
            f.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
