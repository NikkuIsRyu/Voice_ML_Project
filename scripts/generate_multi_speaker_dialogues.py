#!/usr/bin/env python3
"""
Generate Single-File Multi-Speaker Dialogues for Voice-ML v3 (Improvement 3.9).

Features:
1. Role-based multi-speaker assignment (female_v3, male_v1, male_zs) via configs/dialogue_speakers.yaml.
2. Zero GPU-thrashing: Batches turns by voice profile across all dialogues before stitching.
3. Natural acoustic stitching: VAD edge trimming, LUFS loudness matching, room-tone filled pauses,
   and 15ms raised-cosine crossfades.
4. Output: Exactly ONE cohesive WAV per dialogue + JSON alignment timestamps for interactive practice.
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
from typing import Dict, List, Any, Tuple
import yaml
import soundfile as sf
import numpy as np
import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tutor" / "backend"))
from audio_utils import trim_speech

API_URL = "http://127.0.0.1:9880"
GPT_SOVITS_DIR = PROJECT_ROOT / "GPT-SoVITS"

def is_api_alive():
    try:
        r = requests.get(f"{API_URL}/docs", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def ensure_api_running():
    if is_api_alive():
        return None
    print("Starting GPT-SoVITS API server...")
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

def synthesize_turn(text: str, out_path: Path, ref_wav: Path, ref_txt: str, speed: float = 1.0) -> bool:
    payload = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": str(ref_wav.resolve()),
        "prompt_text": ref_txt,
        "prompt_lang": "zh",
        "speed": speed,
        "top_k": 10,
        "top_p": 1.0,
        "temperature": 0.8
    }
    try:
        r = requests.post(f"{API_URL}/tts", json=payload, timeout=120)
        if r.status_code == 200:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True
        else:
            print(f"Synthesis error: {r.status_code} - {r.text}")
            return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def parse_dialogue_file(file_path: Path, speaker_cfg: dict) -> List[Dict[str, Any]]:
    dia_id = file_path.stem
    dia_rules = speaker_cfg.get("dialogues", {}).get(dia_id, {})
    default_rules = speaker_cfg.get("default_mapping", {"A": "female_v3", "B": "male_v1", "C": "male_zs"})
    alternate_list = dia_rules.get("alternate", ["female_v3", "male_v1"])

    lines_out = []
    with open(file_path, "r", encoding="utf-8") as f:
        turn_counter = 0
        for line_idx, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Check for speaker prefix e.g. 'A: ...', 'B: ...', '李先生: ...'
            match = re.match(r"^([A-Za-z\u4e00-\u9fff]+)\s*[:：]\s*(.*)$", line)
            if match:
                spk_label = match.group(1).strip()
                content = match.group(2).strip()
            else:
                spk_label = chr(ord("A") + (turn_counter % 2))
                content = line

            if not content:
                continue

            # Resolve voice profile
            if spk_label in dia_rules:
                voice = dia_rules[spk_label]
            elif spk_label in default_rules:
                voice = default_rules[spk_label]
            elif "alternate" in dia_rules:
                voice = alternate_list[turn_counter % len(alternate_list)]
            else:
                voice = "female_v3" if (turn_counter % 2 == 0) else "male_v1"

            lines_out.append({
                "dialogue_id": dia_id,
                "turn_idx": turn_counter + 1,
                "speaker_label": spk_label,
                "voice_profile": voice,
                "text": content
            })
            turn_counter += 1

    return lines_out

def generate_room_tone(duration_s: float, sr: int = 32000, level_db: float = -65.0) -> np.ndarray:
    """Generate subtle shaped room tone (pinkish Gaussian noise at -65 dBFS)."""
    n_samples = int(duration_s * sr)
    if n_samples <= 0:
        return np.array([], dtype=np.float32)
    # White noise filtered
    np.random.seed(42)
    noise = np.random.randn(n_samples).astype(np.float32)
    # Low-pass filter noise around 1200 Hz for warm room tone
    from scipy import signal
    sos = signal.butter(2, 1200.0, btype="lowpass", fs=sr, output="sos")
    shaped = signal.sosfiltfilt(sos, noise)
    target_rms = 10.0 ** (level_db / 20.0)
    current_rms = np.sqrt(np.mean(shaped**2)) + 1e-9
    return (shaped * (target_rms / current_rms)).astype(np.float32)

def stitch_dialogue_audio(turns_meta: List[Dict[str, Any]], tmp_dir: Path, out_wav: Path, out_json: Path, target_sr: int = 32000):
    """
    Stitch individual turns into a cohesive audio track with room tone pauses and smooth fades.
    """
    audio_segments = []
    timestamps = []
    current_time_ms = 0.0

    fade_samples = int(0.015 * target_sr) # 15ms raised-cosine crossfade
    fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))
    fade_out = 0.5 * (1.0 - np.cos(np.linspace(np.pi, 0, fade_samples)))

    prev_spk = None
    prev_text = ""

    for i, turn in enumerate(turns_meta):
        wav_file = tmp_dir / f"{turn['dialogue_id']}_turn_{turn['turn_idx']:03d}.wav"
        if not wav_file.exists():
            continue

        data, sr = sf.read(str(wav_file))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = data.astype(np.float32)

        # 1. Resample to target_sr if needed
        if sr != target_sr:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # 2. VAD trimming of raw synth edge silence
        trimmed, _ = trim_speech(data, sr=target_sr, pad_ms=30.0)
        if len(trimmed) == 0:
            trimmed = data

        # 3. LUFS / RMS normalization to uniform speech level (-20 dBFS RMS)
        rms = np.sqrt(np.mean(trimmed**2)) + 1e-9
        target_rms = 10.0 ** (-20.0 / 20.0) # ~0.10
        trimmed = trimmed * (target_rms / rms)
        # Peak ceiling at -1.0 dBFS
        peak = np.max(np.abs(trimmed))
        if peak > 0.891:
            trimmed = trimmed * (0.891 / peak)

        # Apply 15ms smooth fade in/out to speech edges
        if len(trimmed) > fade_samples * 2:
            trimmed[:fade_samples] *= fade_in
            trimmed[-fade_samples:] *= fade_out

        # 4. Compute acoustic pause duration before this turn
        if i > 0:
            if turn["speaker_label"] == prev_spk:
                pause_s = 0.30  # 300ms same speaker
            elif any(prev_text.endswith(q) for q in ["？", "?", "吗", "呢"]):
                pause_s = 0.62  # 620ms after question
            else:
                pause_s = 0.52  # 520ms after statement

            room_tone = generate_room_tone(pause_s, sr=target_sr)
            audio_segments.append(room_tone)
            current_time_ms += pause_s * 1000.0

        # Turn duration and timestamp recording
        turn_dur_ms = (len(trimmed) / target_sr) * 1000.0
        start_ms = round(current_time_ms, 1)
        end_ms = round(current_time_ms + turn_dur_ms, 1)

        timestamps.append({
            "turn_idx": turn["turn_idx"],
            "speaker": turn["speaker_label"],
            "voice": turn["voice_profile"],
            "text": turn["text"],
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": round(turn_dur_ms, 1)
        })

        audio_segments.append(trimmed)
        current_time_ms += turn_dur_ms
        prev_spk = turn["speaker_label"]
        prev_text = turn["text"]

    if not audio_segments:
        return

    full_dialogue_audio = np.concatenate(audio_segments)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), full_dialogue_audio, target_sr)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "dialogue_id": turns_meta[0]["dialogue_id"],
            "total_duration_s": round(len(full_dialogue_audio) / target_sr, 2),
            "sample_rate": target_sr,
            "turns_count": len(timestamps),
            "turns": timestamps
        }, f, ensure_ascii=False, indent=2)

    print(f"  -> Generated Single-File Dialogue: {out_wav} ({len(full_dialogue_audio)/target_sr:.1f}s)")
    print(f"  -> Generated Alignment Metadata: {out_json}")

def main():
    parser = argparse.ArgumentParser(description="Synthesize cohesive single-file multi-speaker dialogues.")
    parser.add_argument("--dialogues_dir", default="data/dialogues", help="Directory of dialogue .txt files")
    parser.add_argument("--output_dir", default="output_v2/dialogues", help="Output directory for full dialogues")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier (default: 1.0)")
    parser.add_argument("--generate_slow", action="store_true", help="Also generate 0.75x slow learning dialogue in output_v2_slow")
    parser.add_argument("--dialogues", nargs="+", help="Specific dialogue IDs to synthesize (e.g. Dialogue5_1 Dialogue5_2 ...)")
    args = parser.parse_args()

    dia_dir = (PROJECT_ROOT / args.dialogues_dir).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    tmp_dir = PROJECT_ROOT / "temp_turns"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load configs
    with open(PROJECT_ROOT / "configs" / "voice_profiles.yaml", "r", encoding="utf-8") as f:
        voice_cfg = yaml.safe_load(f)["voices"]
    with open(PROJECT_ROOT / "configs" / "dialogue_speakers.yaml", "r", encoding="utf-8") as f:
        speaker_cfg = yaml.safe_load(f)

    # 2. Parse all dialogues and collect turns
    dialogue_files = sorted(dia_dir.glob("Dialogue*.txt"))
    if args.dialogues:
        dialogue_files = [df for df in dialogue_files if df.stem in args.dialogues]
    all_turns = []
    turns_by_dialogue = {}

    for df in dialogue_files:
        turns = parse_dialogue_file(df, speaker_cfg)
        if turns:
            turns_by_dialogue[df.stem] = turns
            all_turns.extend(turns)

    print(f"Parsed {len(turns_by_dialogue)} dialogues with {len(all_turns)} total conversational turns.")

    # 3. Ensure API server is alive
    proc = ensure_api_running()

    # 4. BATCH SYNTHESIS BY VOICE (ZERO GPU THRASHING)
    # Group turns by voice profile: female_v3 first, then male_v1, then male_zs
    turns_by_voice = {}
    for t in all_turns:
        v = t["voice_profile"]
        turns_by_voice.setdefault(v, []).append(t)

    speeds_to_run = [(args.speed, out_dir)]
    if args.generate_slow:
        slow_out_dir = PROJECT_ROOT / "output_v2_slow" / "dialogues"
        speeds_to_run.append((0.75, slow_out_dir))

    for current_speed, current_out_dir in speeds_to_run:
        print(f"\n{'='*70}")
        print(f"SYNTHESIZING DIALOGUES AT SPEED {current_speed}x (Target: {current_out_dir.name})")
        print(f"{'='*70}")

        # Batch synthesize each voice group
        for voice_name, voice_turns in turns_by_voice.items():
            if voice_name not in voice_cfg:
                print(f"Warning: Unknown voice {voice_name}, defaulting to female_v3")
                voice_name = "female_v3"
            v_info = voice_cfg[voice_name]

            print(f"\n--- Loading Voice Profile: {voice_name} ({len(voice_turns)} turns) ---")
            set_voice_weights(v_info["sovits_weights"], v_info["gpt_weights"])

            ref_wav = PROJECT_ROOT / v_info["ref_audio"]
            ref_txt = v_info["prompt_text"]

            for t in voice_turns:
                turn_tmp_wav = tmp_dir / f"{t['dialogue_id']}_turn_{t['turn_idx']:03d}.wav"
                print(f"[{t['dialogue_id']}: Turn {t['turn_idx']}] ({t['speaker_label']} - {voice_name}): {t['text'][:15]}...")
                synthesize_turn(t["text"], turn_tmp_wav, ref_wav=ref_wav, ref_txt=ref_txt, speed=current_speed)

        # 5. Acoustic stitching per dialogue
        print(f"\n--- Stitching Cohesive Single-File Dialogues ({current_speed}x) ---")
        for dia_id, turns in turns_by_dialogue.items():
            out_wav = current_out_dir / f"{dia_id}.wav"
            out_json = current_out_dir / f"{dia_id}_timestamps.json"
            stitch_dialogue_audio(turns, tmp_dir, out_wav, out_json)

    # Clean up temp turns directory
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nTemporary turns cleaned up. Dialogue synthesis complete!")

if __name__ == "__main__":
    main()
