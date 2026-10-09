#!/usr/bin/env python3
"""
Generate test fixtures for Phase A7:
- Browser formats (.webm, .ogg, .mp4) using ffmpeg
- Short/silent audio to test validation
- Minimal-pair tone audio clips:
  * 买 (mai3) vs 卖 (mai4)
  * 妈 (ma1) vs 骂 (ma4)
  * 是 (shi4) vs 十 (shi2)
  * 四 (si4) vs 十 (shi2)
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_BIN = PROJECT_ROOT / "GPT-SoVITS" / "ffmpeg.exe"
CSMSC_WAV_DIR = PROJECT_ROOT.parent / "Voice_models" / "datasets" / "BZNSYP" / "Wave"
V2_VOCAB_DIR = PROJECT_ROOT / "output_v2" / "vocab"

def run_ffmpeg(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        print(f"FFmpeg error: {p.stderr.decode('utf-8', errors='ignore')}")
    return p.returncode == 0

def slice_and_save(wav_id: str, t_start: float, t_end: float, out_name: str, pad: float = 0.06):
    wav_path = CSMSC_WAV_DIR / f"{wav_id}.wav"
    y, sr = sf.read(str(wav_path))
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    i0 = max(0, int((t_start - pad) * sr))
    i1 = min(len(y), int((t_end + pad) * sr))
    seg = y[i0:i1].astype(np.float32)

    # Normalize peak to 0.9
    peak = np.max(np.abs(seg))
    if peak > 0:
        seg = seg * (0.9 / peak)

    out_path = FIXTURES_DIR / out_name
    sf.write(str(out_path), seg, sr)
    print(f"Created fixture: {out_path} ({len(seg)/sr:.2f}s)")
    return out_path

def main():
    print(f"Preparing test fixtures in {FIXTURES_DIR}...")

    # 1. Slice minimal pairs from CSMSC
    slice_and_save("000028", 0.7788, 0.9925, "mai3_csmsc.wav")
    slice_and_save("000028", 0.9925, 1.2067, "mai4_csmsc.wav")
    slice_and_save("000168", 0.4500, 0.7232, "ma1_csmsc.wav")
    slice_and_save("000666", 4.2752, 4.5486, "ma4_csmsc.wav")
    slice_and_save("000126", 2.1457, 2.3456, "shi2_csmsc.wav")
    slice_and_save("000246", 1.4765, 1.6325, "si4_csmsc.wav")

    # 2. Native V2 clips
    shi4_src = V2_VOCAB_DIR / "lesson01" / "lesson01_005_是.wav"
    if shi4_src.exists():
        y, sr = sf.read(str(shi4_src))
        sf.write(str(FIXTURES_DIR / "shi4_v2.wav"), y, sr)
        print(f"Copied {shi4_src.name} to shi4_v2.wav")

    nihao_src = V2_VOCAB_DIR / "example_lesson" / "example_lesson_005_你好.wav"
    if not nihao_src.exists():
        nihao_src = list(V2_VOCAB_DIR.rglob("*你好*.wav"))[0]

    sf.write(str(FIXTURES_DIR / "nihao.wav"), *sf.read(str(nihao_src)))

    # 3. Convert nihao.wav to browser formats (.webm, .ogg, .mp4) using ffmpeg
    ffmpeg_str = str(FFMPEG_BIN)
    nihao_wav = str(FIXTURES_DIR / "nihao.wav")

    # WebM Opus
    webm_out = str(FIXTURES_DIR / "nihao.webm")
    run_ffmpeg([ffmpeg_str, "-y", "-i", nihao_wav, "-c:a", "libopus", "-b:a", "32k", webm_out])
    print(f"Created {webm_out}")

    # OGG Vorbis
    ogg_out = str(FIXTURES_DIR / "nihao.ogg")
    run_ffmpeg([ffmpeg_str, "-y", "-i", nihao_wav, "-c:a", "libvorbis", "-q:a", "4", ogg_out])
    print(f"Created {ogg_out}")

    # MP4 AAC
    mp4_out = str(FIXTURES_DIR / "nihao.mp4")
    run_ffmpeg([ffmpeg_str, "-y", "-i", nihao_wav, "-c:a", "aac", "-b:a", "64k", mp4_out])
    print(f"Created {mp4_out}")

    # 4. Generate short/silent clip for validation testing (0.15s of silence)
    silent_audio = np.zeros(int(16000 * 0.15), dtype=np.float32)
    silent_wav = str(FIXTURES_DIR / "silent.wav")
    sf.write(silent_wav, silent_audio, 16000)
    silent_webm = str(FIXTURES_DIR / "silent.webm")
    run_ffmpeg([ffmpeg_str, "-y", "-i", silent_wav, "-c:a", "libopus", silent_webm])
    print(f"Created {silent_webm}")

    print("All fixtures prepared successfully!")

if __name__ == "__main__":
    main()
