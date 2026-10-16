#!/usr/bin/env python3
"""
Evaluate female_v3 checkpoints (Epoch 8, 10, 12), select the winner via CER & Tone Accuracy,
and generate 3 side-by-side comparison samples (female_v2 vs female_v3) for User-Gate G1.
"""

import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import pypinyin
import requests
import soundfile as sf
import numpy as np

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
REF_WAV = PROJECT_ROOT / "data" / "train_v3" / "csmsc" / "csmsc_ref.wav"
REF_TXT = PROJECT_ROOT / "data" / "train_v3" / "csmsc" / "csmsc_ref.txt"

CANDIDATES = [
    {
        "name": "female_v3_epoch8",
        "sovits": "SoVITS_weights_v2ProPlus/female_v3_e8_s4936.pth",
        "gpt": "GPT_weights_v2ProPlus/female_v3-e8.ckpt"
    },
    {
        "name": "female_v3_epoch10",
        "sovits": "SoVITS_weights_v2ProPlus/female_v3_e10_s6170.pth",
        "gpt": "GPT_weights_v2ProPlus/female_v3-e10.ckpt"
    },
    {
        "name": "female_v3_epoch12",
        "sovits": "SoVITS_weights_v2ProPlus/female_v3_e12_s7404.pth",
        "gpt": "GPT_weights_v2ProPlus/female_v3-e12.ckpt"
    }
]

V2_BASELINE = {
    "name": "female_v2",
    "sovits": "SoVITS_weights_v2ProPlus/csmsc_v2_e4_s2000.pth",
    "gpt": "GPT_weights_v2ProPlus/csmsc_v2-e15.ckpt"
}

TEST_ITEMS = [
    ("V01", "您好。"),
    ("V02", "很好。"),
    ("V06", "不是。"),
    ("V11", "妈。"),
    ("V12", "麻。"),
    ("V13", "马。"),
    ("V14", "骂。"),
    ("V15", "中国银行。"),
    ("V29", "图书馆。"),
    ("D01", "您好，请问您是王老师吗？"),
    ("D04", "认识你很高兴，张华！你的汉语说得很好。"),
    ("D07", "我是德国人，我家在柏林。"),
    ("D09", "今天下午你有时间吗？我们可以一起去图书馆借书。"),
    ("T01", "妈妈骑马，马慢，妈妈骂马。"),
    ("T02", "八百标兵奔北坡，炮兵并排北边跑。")
]

GATE_SAMPLES = [
    ("sample1_minimal_pair_100x", "妈妈骑马，马慢，妈妈骂马。", 1.0),
    ("sample2_slow_vocab_065x", "图书馆。认识你很高兴。", 0.65),
    ("sample3_dialogue_100x", "今天下午你有时间吗？我们可以一起去图书馆借书。", 1.0)
]

def strip_punct(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", text)

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
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

def compute_cer(target: str, hyp: str) -> float:
    t_clean = strip_punct(target)
    h_clean = strip_punct(hyp)
    if not t_clean:
        return 0.0 if not h_clean else 1.0
    return min(levenshtein_distance(t_clean, h_clean) / len(t_clean), 1.0)

def compute_tone_accuracy(target: str, hyp: str) -> float:
    t_clean = strip_punct(target)
    h_clean = strip_punct(hyp)
    t_pyn = [p[0] for p in pypinyin.pinyin(t_clean, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)]
    h_pyn = [p[0] for p in pypinyin.pinyin(h_clean, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)]
    if not t_pyn:
        return 1.0
    matches = sum(1 for t, h in zip(t_pyn, h_pyn) if t == h)
    return matches / max(len(t_pyn), len(h_pyn))

def is_api_alive(url="http://127.0.0.1:9880"):
    try:
        r = requests.get(f"{url}/docs", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def start_api():
    if is_api_alive():
        print("TTS API is already running.")
        return None
    print("Launching TTS API server...")
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
            print("TTS API successfully started.")
            return proc
    raise RuntimeError("Timed out waiting for TTS API to start.")

def stop_api(proc):
    if proc:
        print("Stopping TTS API server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        time.sleep(3)

def set_weights(sovits_path: str, gpt_path: str):
    sovits_abs = str((GPT_SOVITS_DIR / sovits_path).resolve())
    gpt_abs = str((GPT_SOVITS_DIR / gpt_path).resolve())
    r1 = requests.get("http://127.0.0.1:9880/set_sovits_weights", params={"weights_path": sovits_abs}, timeout=15)
    r2 = requests.get("http://127.0.0.1:9880/set_gpt_weights", params={"weights_path": gpt_abs}, timeout=15)
    if r1.status_code != 200 or r2.status_code != 200:
        raise RuntimeError(f"Failed to set weights: sovits={r1.text}, gpt={r2.text}")
    print(f"Loaded weights: {Path(sovits_path).name} | {Path(gpt_path).name}")

def synthesize_text(text: str, out_wav: Path, speed: float = 1.0, prompt_text: str = "", prompt_wav: Path = None):
    url = "http://127.0.0.1:9880/tts"
    payload = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": str(prompt_wav.resolve()),
        "prompt_text": prompt_text,
        "prompt_lang": "zh",
        "speed": speed,
        "top_k": 10,
        "top_p": 1.0,
        "temperature": 0.8
    }
    r = requests.post(url, json=payload, timeout=120)
    if r.status_code == 200:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        with open(out_wav, "wb") as f:
            f.write(r.content)
        return True
    else:
        print(f"Error {r.status_code} synthesizing '{text}': {r.text}")
        return False

def main():
    with open(REF_TXT, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    print(f"Prompt Audio: {REF_WAV}")
    print(f"Prompt Text : {prompt_text}")

    proc = start_api()
    eval_dir = PROJECT_ROOT / "eval" / "v3_sweep"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthesize evaluation items for all candidates
    for cand in CANDIDATES:
        print(f"\n--- Synthesizing testset for {cand['name']} ---")
        set_weights(cand["sovits"], cand["gpt"])
        cand_dir = eval_dir / cand["name"]
        cand_dir.mkdir(parents=True, exist_ok=True)
        for item_id, text in TEST_ITEMS:
            wav_path = cand_dir / f"{item_id}.wav"
            synthesize_text(text, wav_path, speed=1.0, prompt_text=prompt_text, prompt_wav=REF_WAV)

    # Stop TTS API before loading Whisper onto GPU to prevent VRAM competition
    stop_api(proc)

    # 2. Evaluate with Faster-Whisper on GPU
    print("\nLoading Faster-Whisper on GPU for evaluation...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(str(ASR_DIR), device="cuda", compute_type="float16")

    candidate_results = []
    for cand in CANDIDATES:
        cand_dir = eval_dir / cand["name"]
        cers = []
        tone_accs = []
        for item_id, target_text in TEST_ITEMS:
            wav_path = cand_dir / f"{item_id}.wav"
            if not wav_path.exists():
                cers.append(1.0)
                tone_accs.append(0.0)
                continue
            segments, _ = whisper_model.transcribe(str(wav_path), language="zh", beam_size=5)
            hyp_text = "".join(s.text for s in segments).strip()
            cer = compute_cer(target_text, hyp_text)
            t_acc = compute_tone_accuracy(target_text, hyp_text)
            cers.append(cer)
            tone_accs.append(t_acc)

        mean_cer = float(np.mean(cers)) * 100.0 if "np" in globals() else sum(cers) / len(cers) * 100.0
        mean_tone = float(np.mean(tone_accs)) * 100.0 if "np" in globals() else sum(tone_accs) / len(tone_accs) * 100.0
        candidate_results.append({
            "cand": cand,
            "cer": mean_cer,
            "tone_acc": mean_tone
        })
        print(f"[{cand['name']}] CER: {mean_cer:.2f}% | Tone Accuracy: {mean_tone:.2f}%")

    # Select best candidate (lowest CER, highest Tone Accuracy)
    candidate_results.sort(key=lambda x: (x["cer"], -x["tone_acc"]))
    best = candidate_results[0]
    print(f"\nWinning Checkpoint: {best['cand']['name']} (CER: {best['cer']:.2f}%, Tone: {best['tone_acc']:.2f}%)")

    # Free Whisper memory
    del whisper_model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Generate side-by-side comparison samples (female_v2 vs female_v3)
    comp_dir = PROJECT_ROOT / "eval" / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    proc = start_api()
    print("\n--- Generating Side-by-Side Comparison Audio for User-Gate G1 ---")

    # Render with female_v2 baseline
    print("Rendering female_v2 baseline...")
    set_weights(V2_BASELINE["sovits"], V2_BASELINE["gpt"])
    # CSMSC reference audio for v2
    v2_ref_wav = PROJECT_ROOT / "data" / "reference_audio" / "A4_137.wav"
    v2_ref_txt = "不弃雅不避俗化俗为雅化雅为俗俗不伤雅雅俗共赏。"
    if not v2_ref_wav.exists():
        v2_ref_wav = REF_WAV
        v2_ref_txt = prompt_text

    for tag, text, spd in GATE_SAMPLES:
        out_f = comp_dir / f"{tag}_v2.wav"
        synthesize_text(text, out_f, speed=spd, prompt_text=v2_ref_txt, prompt_wav=v2_ref_wav)

    # Render with winning female_v3
    print(f"Rendering {best['cand']['name']}...")
    set_weights(best["cand"]["sovits"], best["cand"]["gpt"])
    for tag, text, spd in GATE_SAMPLES:
        out_f = comp_dir / f"{tag}_v3.wav"
        synthesize_text(text, out_f, speed=spd, prompt_text=prompt_text, prompt_wav=REF_WAV)

    stop_api(proc)

    # Save summary report
    summary_path = comp_dir / "gate_g1_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "winning_checkpoint": best["cand"],
            "winning_cer": best["cer"],
            "winning_tone_acc": best["tone_acc"],
            "all_sweep_results": candidate_results,
            "gate_samples": [
                {
                    "tag": tag,
                    "text": text,
                    "speed": spd,
                    "v2_path": str(comp_dir / f"{tag}_v2.wav").replace("\\", "/"),
                    "v3_path": str(comp_dir / f"{tag}_v3.wav").replace("\\", "/")
                }
                for tag, text, spd in GATE_SAMPLES
            ]
        }, f, ensure_ascii=False, indent=2)

    print(f"\nGate G1 comparison samples ready in {comp_dir}!")

if __name__ == "__main__":
    main()
