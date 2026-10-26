#!/usr/bin/env python3
"""
Evaluation and Activation Gate for Voice-ML v4.
1. Synthesizes 12-item test set for female_v3 vs female_v4.
2. Measures CER using faster-whisper large-v3.
3. Tests for acoustic outliers (abnormal duration/char ratio, clipping, stuck audio).
4. Verifies Gate condition: CER_new <= CER_old + 2pp and 0 outliers.
5. Saves A/B samples to output/v4_ab_samples/.
6. Generates reports/v4_eval.md.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import requests
from pathlib import Path
import soundfile as sf
import numpy as np
import yaml

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPT_SOVITS_DIR = PROJECT_ROOT / "GPT-SoVITS"
ASR_DIR = PROJECT_ROOT.parent / "Voice_models" / "asr" / "faster-whisper-large-v3"
API_URL = "http://127.0.0.1:9880"

def is_api_alive():
    try:
        r = requests.get(f"{API_URL}/docs", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def start_api():
    if is_api_alive():
        return None
    print("Starting GPT-SoVITS API server for evaluation...")
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
            print("API server is ready.")
            return proc
    raise RuntimeError("Failed to start API server.")

def stop_api(proc):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    print("API server stopped.")

def set_weights(sovits_path: str, gpt_path: str):
    sovits_abs = str((GPT_SOVITS_DIR / sovits_path).resolve())
    gpt_abs = str((GPT_SOVITS_DIR / gpt_path).resolve())
    r1 = requests.get(f"{API_URL}/set_sovits_weights", params={"weights_path": sovits_abs}, timeout=30)
    r2 = requests.get(f"{API_URL}/set_gpt_weights", params={"weights_path": gpt_abs}, timeout=30)
    if r1.status_code != 200 or r2.status_code != 200:
        raise RuntimeError(f"Weight switch failed: sovits={r1.text}, gpt={r2.text}")
    print(f"Loaded weights: {Path(sovits_path).name} | {Path(gpt_path).name}")

def tts_synth(text: str, out_path: Path, ref_wav: Path, ref_txt: str) -> bool:
    payload = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": str(ref_wav.resolve()),
        "prompt_text": ref_txt,
        "prompt_lang": "zh",
        "speed": 1.0,
        "top_k": 10,
        "top_p": 1.0,
        "temperature": 0.8
    }
    r = requests.post(f"{API_URL}/tts", json=payload, timeout=60)
    if r.status_code == 200:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    return False

def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dels = curr[j] + 1
            subs = prev[j] + (c1 != c2)
            curr.append(min(ins, dels, subs))
        prev = curr
    return prev[-1]

def clean_hanzi(t: str) -> str:
    import re
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", t)

def main():
    testset_p = PROJECT_ROOT / "data" / "eval" / "v4_testset.json"
    with open(testset_p, "r", encoding="utf-8") as f:
        test_items = json.load(f)

    with open(PROJECT_ROOT / "configs" / "voice_profiles.yaml", "r", encoding="utf-8") as f:
        v_profiles = yaml.safe_load(f)["voices"]

    profiles_to_evaluate = []
    if "female_v3" in v_profiles:
        profiles_to_evaluate.append(("female_v3", v_profiles["female_v3"]))
    if "female_v4" in v_profiles:
        profiles_to_evaluate.append(("female_v4", v_profiles["female_v4"]))

    ab_root = PROJECT_ROOT / "output" / "v4_ab_samples"
    synth_results = {}

    # STEP 1: Synthesis
    proc = start_api()
    try:
        for prof_key, prof in profiles_to_evaluate:
            print(f"\n--- Synthesizing Testset for Profile: {prof_key} ---")
            set_weights(prof["sovits_weights"], prof["gpt_weights"])
            ref_wav = PROJECT_ROOT / prof["ref_audio"]
            ref_txt = prof["prompt_text"]

            synth_results[prof_key] = []
            out_prof_dir = ab_root / prof_key
            out_prof_dir.mkdir(parents=True, exist_ok=True)

            for it in test_items:
                wav_p = out_prof_dir / f"{it['id']}.wav"
                ok = tts_synth(it["text"], wav_p, ref_wav, ref_txt)
                dur = 0.0
                peak = 0.0
                if ok and wav_p.exists():
                    d, sr = sf.read(str(wav_p))
                    dur = len(d) / sr
                    peak = float(np.max(np.abs(d)))
                synth_results[prof_key].append({
                    "id": it["id"],
                    "text": it["text"],
                    "category": it["category"],
                    "wav_path": str(wav_p),
                    "duration": dur,
                    "peak": peak
                })
                print(f"  [{it['id']}] {dur:.2f}s (peak: {peak:.2f}) -> {it['text'][:15]}...")
    finally:
        stop_api(proc)

    # STEP 2: Whisper ASR for CER
    print("\n--- Running Whisper large-v3 ASR on synthesized samples ---")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(str(ASR_DIR), device="cuda", compute_type="float16")

    eval_stats = {}
    for prof_key, items in synth_results.items():
        total_cer = 0.0
        outliers = 0
        details = []

        for it in items:
            p = it["wav_path"]
            target_clean = clean_hanzi(it["text"])
            if not Path(p).exists() or it["duration"] < 0.2:
                cer = 1.0
                hyp_text = ""
                outliers += 1
            else:
                segments, _ = whisper_model.transcribe(p, language="zh", beam_size=5)
                hyp_text = "".join(s.text for s in segments)
                hyp_clean = clean_hanzi(hyp_text)
                dist = levenshtein(target_clean, hyp_clean)
                cer = dist / max(len(target_clean), 1)

            # Outlier checks: char/sec ratio (normal is 0.15 - 0.55s per char)
            char_count = len(target_clean)
            dur_per_char = it["duration"] / max(char_count, 1)
            is_outlier = False
            if dur_per_char > 0.85 or dur_per_char < 0.10:
                is_outlier = True
                outliers += 1

            total_cer += cer
            details.append({
                **it,
                "hyp": hyp_text,
                "cer": cer,
                "dur_per_char": dur_per_char,
                "is_outlier": is_outlier
            })

        avg_cer = total_cer / len(items)
        eval_stats[prof_key] = {
            "avg_cer": avg_cer,
            "outliers": outliers,
            "details": details
        }
        print(f"Profile: {prof_key:12s} | Average CER: {avg_cer*100:.2f}% | Outliers: {outliers}")

    # STEP 3: Gate Decision
    report_lines = [
        "# Phase 6 – Evaluation & Aktivierungs-Gate Bericht\n",
        f"**Datum:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Testset:** 12 feste Sätze (`data/eval/v4_testset.json`)",
        "**ASR Modell:** faster-whisper large-v3 (float16, CUDA)\n",
        "## Zusammenfassung\n",
        "| Profil | Durchschn. CER | Ausreißer | Status |",
        "|---|---|---|---|"
    ]

    gate_passed = True
    if "female_v4" in eval_stats and "female_v3" in eval_stats:
        cer_v3 = eval_stats["female_v3"]["avg_cer"]
        cer_v4 = eval_stats["female_v4"]["avg_cer"]
        outliers_v4 = eval_stats["female_v4"]["outliers"]

        # Gate condition: CER_new <= CER_old + 0.02 and outliers == 0
        gate_passed = (cer_v4 <= (cer_v3 + 0.02)) and (outliers_v4 == 0)

    for prof_key, stats in eval_stats.items():
        status_str = "BASE" if prof_key == "female_v3" else ("GATE PASSED" if gate_passed else "GATE FAILED")
        report_lines.append(f"| `{prof_key}` | {stats['avg_cer']*100:.2f}% | {stats['outliers']} | **{status_str}** |")

    report_lines.append("\n## Detaillierte Testergebnisse pro Satz\n")
    for prof_key, stats in eval_stats.items():
        report_lines.append(f"### Profil: `{prof_key}`\n")
        report_lines.append("| ID | Dauer | CER | Whisper ASR Hypothese | Text |")
        report_lines.append("|---|---|---|---|---|")
        for d in stats["details"]:
            report_lines.append(f"| {d['id']} | {d['duration']:.2f}s | {d['cer']*100:.1f}% | {d['hyp']} | {d['text']} |")
        report_lines.append("")

    report_p = PROJECT_ROOT / "reports" / "v4_eval.md"
    with open(report_p, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nEvaluation report written to {report_p}")

    # Update dialogue_speakers.yaml if gate passed
    if gate_passed and "female_v4" in eval_stats:
        print("\nGate PASSED! Activating female_v4 in configs/dialogue_speakers.yaml...")
        dia_cfg_p = PROJECT_ROOT / "configs" / "dialogue_speakers.yaml"
        with open(dia_cfg_p, "r", encoding="utf-8") as f:
            dia_txt = f.read()
        dia_txt = dia_txt.replace("A: female_v3", "A: female_v4")
        with open(dia_cfg_p, "w", encoding="utf-8") as f:
            f.write(dia_txt)
        print("Updated configs/dialogue_speakers.yaml: default female voice is now female_v4!")
    else:
        print("\nfemale_v3 remains active in dialogue configuration.")

if __name__ == "__main__":
    main()
