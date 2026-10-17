#!/usr/bin/env python3
"""
Resilient streaming extractor for AISHELL-3 male speakers.
Features:
- Seamless HTTP range reconnection on network timeout (never loses progress).
- Target male speakers: SSB0863, SSB0817, SSB1187, SSB1239, SSB1328, SSB1365, SSB1630.
- Extracts WAV files directly to data/train_v3/aishell3_male/wavs/.
- Parses transcripts from local data/aishell3_meta/test/content.txt.
"""

import io
import os
import re
import sys
import time
import tarfile
from pathlib import Path
import requests
import soundfile as sf
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tutor" / "backend"))
from audio_utils import trim_speech

URL = "https://openslr.trmal.net/resources/93/data_aishell3.tgz"
OUT_DIR = PROJECT_ROOT / "data" / "train_v3" / "aishell3_male"
WAV_DIR = OUT_DIR / "wavs"
REF_DIR = PROJECT_ROOT / "data" / "refs" / "male_zs"

TARGET_MALE_SPKS = {"SSB0863", "SSB0817", "SSB1187", "SSB1239", "SSB1328", "SSB1365", "SSB1630"}

class ResilientHttpStream(io.RawIOBase):
    """File-like stream that auto-reconnects with Range header on network drops."""
    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.resp = None
        self.iter = None
        self._buf = bytearray()
        self._connect()

    def _connect(self):
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Range": f"bytes={self.pos}-"
        }
        for attempt in range(10):
            try:
                self.resp = requests.get(self.url, headers=headers, stream=True, timeout=30)
                if self.resp.status_code in [200, 206]:
                    self.iter = self.resp.iter_content(chunk_size=1048576)
                    return
            except Exception as e:
                print(f"Connection attempt {attempt+1} failed: {e}. Retrying in 3s...")
                time.sleep(3)
        raise RuntimeError("Failed to connect after 10 attempts")

    def readable(self):
        return True

    def readinto(self, b):
        n = len(b)
        while len(self._buf) < n:
            try:
                chunk = next(self.iter)
                self._buf.extend(chunk)
                self.pos += len(chunk)
            except StopIteration:
                break
            except Exception as e:
                print(f"\n[Warning] Network interruption at byte {self.pos}: {e}. Auto-reconnecting...")
                self._connect()
                continue

        to_copy = min(n, len(self._buf))
        b[:to_copy] = self._buf[:to_copy]
        del self._buf[:to_copy]
        return to_copy

def main():
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load transcripts from test/content.txt
    content_txt = PROJECT_ROOT / "data" / "aishell3_meta" / "test" / "content.txt"
    if not content_txt.exists():
        raise FileNotFoundError(f"Missing {content_txt}")

    transcripts = {}
    with open(content_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                wav_name = parts[0].strip()
                # Text format: '武 wu3 术 shu4 始 shi3 ...'
                tokens = parts[1].strip().split()
                # Keep Hanzi characters only
                hanzi_chars = [c for c in tokens if re.match(r"^[\u4e00-\u9fff]$", c)]
                hanzi_text = "".join(hanzi_chars)
                if hanzi_text and not any(hanzi_text.endswith(p) for p in "。？！"):
                    hanzi_text += "。"
                transcripts[wav_name] = hanzi_text

    print(f"Loaded {len(transcripts)} transcripts from {content_txt.name}")

    # 2. Resilient streaming extraction from OpenSLR 93
    print(f"\nConnecting resilient HTTP stream to OpenSLR 93 (auto-reconnect enabled)...")
    raw_stream = ResilientHttpStream(URL)

    extracted_counts = {spk: 0 for spk in TARGET_MALE_SPKS}
    manifest_entries = []
    male_zs_candidates = []
    total_male_v1_dur = 0.0

    target_v1_seconds = 2.5 * 3600.0 # 2.5 hours (~9000s)

    with tarfile.open(fileobj=raw_stream, mode="r|gz") as tar:
        for member in tar:
            parts = member.name.replace("\\", "/").split("/")
            if len(parts) >= 3 and parts[0] == "test" and parts[1] == "wav":
                spk = parts[2]
                wav_name = Path(member.name).name
                if spk in TARGET_MALE_SPKS and wav_name.endswith(".wav"):
                    if wav_name not in transcripts:
                        continue
                    hanzi = transcripts[wav_name]

                    # Extract file in memory
                    fobj = tar.extractfile(member)
                    if not fobj:
                        continue
                    audio_bytes = fobj.read()
                    data, sr = sf.read(io.BytesIO(audio_bytes))
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    data = data.astype(np.float32)

                    # VAD trim
                    trimmed, vad_meta = trim_speech(data, sr=sr)
                    dur = len(trimmed) / sr
                    if dur < 1.8 or dur > 9.0 or vad_meta["speech_ratio"] < 0.45:
                        continue

                    # Resample to 32kHz or keep 44.1kHz (GPT-SoVITS pipeline standardizes)
                    # Normalize peak to -1.5 dBFS
                    pk = float(np.max(np.abs(trimmed)))
                    if pk > 1e-4:
                        trimmed = trimmed * (0.841 / pk)

                    # Separate male_zs (SSB1630 is deeper voice for Speaker C in 3-person dialogues)
                    if spk == "SSB1630":
                        if len(male_zs_candidates) < 15:
                            zs_path = REF_DIR / wav_name
                            sf.write(str(zs_path), trimmed, sr)
                            zs_wav_str = str(zs_path).replace("\\", "/")
                            male_zs_candidates.append({
                                "wav": zs_wav_str,
                                "dur": dur,
                                "text": hanzi
                            })
                            extracted_counts[spk] += 1
                        continue

                    # male_v1 training corpus
                    if total_male_v1_dur < target_v1_seconds:
                        out_wav = WAV_DIR / wav_name
                        sf.write(str(out_wav), trimmed, sr)
                        out_wav_str = str(out_wav).replace("\\", "/")
                        manifest_entries.append(f"{out_wav_str}|male_v1|zh|{hanzi}")
                        total_male_v1_dur += dur
                        extracted_counts[spk] += 1

                        if len(manifest_entries) % 100 == 0:
                            print(f"[{len(manifest_entries)} clips] male_v1: {total_male_v1_dur/60:.1f}m ({total_male_v1_dur/3600:.2f}h) | Stream pos: {raw_stream.pos/1024/1024:.1f}MB")

            if total_male_v1_dur >= target_v1_seconds and len(male_zs_candidates) >= 10:
                print(f"\nReached target duration for male_v1 ({total_male_v1_dur/3600:.2f}h) and male_zs ({len(male_zs_candidates)} clips). Cleanly terminating stream!")
                break

            if member.name.startswith("train/"):
                print("Reached train/ directory in archive.")
                break

    # 3. Write manifests
    name2text_file = OUT_DIR / "2-name2text.txt"
    with open(name2text_file, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_entries) + "\n")
    print(f"\nWrote male_v1 training manifest: {name2text_file} ({len(manifest_entries)} clips)")

    # Select best prompt reference audio for male_v1
    ref_candidates = [
        e.split("|") for e in manifest_entries
        if 4.5 <= (len(sf.read(e.split("|")[0])[0]) / sf.read(e.split("|")[0])[1]) <= 6.5
    ]
    best_ref = ref_candidates[0] if ref_candidates else manifest_entries[0].split("|")
    ref_wav_out = OUT_DIR / "male_v1_ref.wav"
    ref_txt_out = OUT_DIR / "male_v1_ref.txt"

    d_ref, sr_ref = sf.read(best_ref[0])
    sf.write(str(ref_wav_out), d_ref, sr_ref)
    with open(ref_txt_out, "w", encoding="utf-8") as f:
        f.write(best_ref[3])
    print(f"Selected male_v1 prompt reference: {ref_wav_out} ('{best_ref[3]}')")

    # Select best zero-shot prompt reference for male_zs
    male_zs_candidates.sort(key=lambda x: abs(x["dur"] - 5.5))
    best_zs = male_zs_candidates[0]
    zs_ref_wav = REF_DIR / "male_zs_ref.wav"
    zs_ref_txt = REF_DIR / "male_zs_ref.txt"

    d_zs, sr_zs = sf.read(best_zs["wav"])
    sf.write(str(zs_ref_wav), d_zs, sr_zs)
    with open(zs_ref_txt, "w", encoding="utf-8") as f:
        f.write(best_zs["text"])
    print(f"Selected male_zs zero-shot prompt reference: {zs_ref_wav} ('{best_zs['text']}')")

    print(f"\nExtraction summary:")
    print(f"  male_v1 clips: {len(manifest_entries)} ({total_male_v1_dur/3600:.2f} hours)")
    print(f"  male_zs clips: {len(male_zs_candidates)}")
    print(f"  Downloaded stream size: {raw_stream.pos/1024/1024:.1f} MB (saved ~17 GB download!)")

if __name__ == "__main__":
    main()
