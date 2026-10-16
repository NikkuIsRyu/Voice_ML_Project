import os
import sys
import re
import json
import argparse
from pathlib import Path
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

def clean_prosody_text(raw_text: str) -> str:
    """
    Convert DataBaker CSMSC prosodic tags to standard Chinese punctuation:
    #1: minor word boundary (remove)
    #2: minor phrase pause (replace with comma if between words)
    #3: major phrase boundary (comma '，')
    #4: sentence boundary (period '。')
    """
    t = raw_text.strip()
    t = t.replace("#3", "，")
    t = t.replace("#4", "。")
    t = t.replace("#2", "，")
    t = t.replace("#1", "")
    
    t = re.sub(r"[，,]+", "，", t)
    t = re.sub(r"[。.]+", "。", t)
    t = re.sub(r"[，,][。.]", "。", t)
    t = re.sub(r"^[，。？！、\s]+", "", t)
    t = t.strip()
    if t and not any(t.endswith(p) for p in "。？！…"):
        t += "。"
    return t

def normalize_loudness(audio_data: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    """
    RMS-based loudness normalization to approximate -23 LUFS with peak limiting.
    """
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms < 1e-6:
        return audio_data
    target_rms = 10.0 ** (target_lufs / 20.0)
    gain = target_rms / rms
    norm = audio_data * gain
    peak = np.max(np.abs(norm))
    if peak > 0.891:
        norm = norm * (0.891 / peak)
    return norm

def main():
    parser = argparse.ArgumentParser(description="Prepare Biaobei CSMSC dataset for GPT-SoVITS")
    parser.add_argument("--source_dir", default="../Voice_models/datasets/BZNSYP",
                        help="Path to unpacked BZNSYP folder")
    parser.add_argument("--output_dir", default="data/train_v3/csmsc",
                        help="Output directory for formatted dataset")
    parser.add_argument("--target_hours", type=float, default=2.5,
                        help="Target total duration in hours (default: 2.5 hours = ~9000s)")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Maximum number of sentences to prepare (optional)")
    parser.add_argument("--holdout", type=int, default=100,
                        help="Number of clean items reserved in holdout directory for eval/reference selection")
    parser.add_argument("--min_dur", type=float, default=2.0,
                        help="Minimum clip duration in seconds (default: 2.0)")
    parser.add_argument("--max_dur", type=float, default=10.0,
                        help="Maximum clip duration in seconds (default: 10.0)")
    args = parser.parse_args()

    src = Path(args.source_dir).resolve()
    out = Path(args.output_dir).resolve()
    wav_out = out / "wavs"
    wav_out.mkdir(parents=True, exist_ok=True)

    holdout_out = out.parent / f"{out.name}_holdout"
    holdout_wav_out = holdout_out / "wavs"
    if args.holdout > 0:
        holdout_wav_out.mkdir(parents=True, exist_ok=True)

    # Locate ProsodyLabeling file
    label_files = list(src.glob("**/000001-010000.txt"))
    if not label_files:
        raise FileNotFoundError(f"Could not find 000001-010000.txt in {src}")
    label_path = label_files[0]
    print(f"Found labeling file: {label_path}")

    # Locate Wave directory
    wave_dirs = list(src.glob("**/Wave"))
    if not wave_dirs:
        wav_candidates = list(src.glob("**/*.wav"))
        if not wav_candidates:
            raise FileNotFoundError(f"Could not find Wave directory or WAV files in {src}")
        wave_dir = wav_candidates[0].parent
    else:
        wave_dir = wave_dirs[0]
    print(f"Found Wave directory: {wave_dir}")

    # Read and parse labels
    items = []
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for i in range(0, len(lines), 2):
        parts = lines[i].split("\t")
        if len(parts) >= 2:
            wav_id = parts[0].strip()
            raw_text = parts[1].strip()
            clean_text = clean_prosody_text(raw_text)
            pinyin = lines[i+1].strip() if i+1 < len(lines) else ""
            items.append({
                "id": wav_id,
                "raw_text": raw_text,
                "text": clean_text,
                "pinyin": pinyin
            })

    print(f"Parsed {len(items)} items from CSMSC transcript.")

    target_seconds = (args.target_hours * 3600.0) if args.target_hours else float("inf")
    print(f"Target duration: {args.target_hours:.2f}h ({target_seconds:.0f}s), holdout: {args.holdout} clips")

    holdout_items = []
    selected = []
    total_duration = 0.0
    rejected_quality = 0

    print(f"\nProcessing and curating audio files (serially, low CPU footprint)...")
    for item in items:
        if args.max_items and len(selected) >= args.max_items:
            break
        if total_duration >= target_seconds:
            break

        wav_name = f"{item['id']}.wav"
        src_wav = wave_dir / wav_name
        if not src_wav.exists():
            continue

        try:
            data, sr = sf.read(str(src_wav))
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            data = data.astype(np.float32)

            # Quality gate 1: Reject clipping
            if np.max(np.abs(data)) >= 0.999:
                rejected_quality += 1
                continue

            # Quality gate 2: VAD trimming of edge silence
            trimmed_data, vad_meta = trim_speech(data, sr=sr, top_db=30.0)
            dur = len(trimmed_data) / sr

            # Duration boundaries
            if dur < args.min_dur or dur > args.max_dur:
                continue

            # Quality gate 3: Reject abnormal silence / low active speech ratio
            if vad_meta["speech_ratio"] < 0.50:
                rejected_quality += 1
                continue

            # Loudness normalization
            norm_data = normalize_loudness(trimmed_data, sr, target_lufs=-23.0)

            # Deterministic holdout split: every 30th clean candidate until holdout target reached
            if len(holdout_items) < args.holdout and (len(selected) + len(holdout_items)) % 25 == 0:
                h_wav = holdout_wav_out / wav_name
                sf.write(str(h_wav), norm_data, sr)
                holdout_items.append({
                    **item,
                    "wav_path": str(h_wav).replace("\\", "/"),
                    "duration": dur
                })
                continue

            # Save training wav
            tgt_wav = wav_out / wav_name
            sf.write(str(tgt_wav), norm_data, sr)

            selected.append({
                **item,
                "wav_path": str(tgt_wav).replace("\\", "/"),
                "duration": dur
            })
            total_duration += dur

            if len(selected) % 200 == 0:
                print(f"[{len(selected)} clips] Total: {total_duration/60:.1f} min ({total_duration/3600:.2f} h) | Holdout: {len(holdout_items)}")

        except Exception as e:
            print(f"Error processing {src_wav}: {e}")

    print(f"\nSelection complete:")
    print(f"  Training clips : {len(selected)} ({total_duration/3600:.2f} hours)")
    print(f"  Holdout clips  : {len(holdout_items)}")
    print(f"  Rejected clips : {rejected_quality}")

    # Write training list for GPT-SoVITS
    name2text_path = out / "2-name2text.txt"
    with open(name2text_path, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(f"{item['wav_path']}|csmsc|zh|{item['text']}\n")
    print(f"Wrote training manifest: {name2text_path}")

    # Write holdout manifest
    if holdout_items:
        holdout_manifest_path = holdout_out / "holdout_manifest.json"
        with open(holdout_manifest_path, "w", encoding="utf-8") as f:
            json.dump(holdout_items, f, ensure_ascii=False, indent=2)
        print(f"Wrote holdout manifest: {holdout_manifest_path}")

    # Select best reference audio for prompting (5.0s to 7.0s, natural length, clear text)
    ref_candidates = [
        item for item in (holdout_items + selected)
        if 5.0 <= item["duration"] <= 7.0 and 12 <= len(item["text"]) <= 22
    ]
    ref_item = ref_candidates[0] if ref_candidates else selected[0]

    ref_wav_out = out / "csmsc_ref.wav"
    ref_txt_out = out / "csmsc_ref.txt"

    data, sr = sf.read(ref_item["wav_path"])
    sf.write(str(ref_wav_out), data, sr)
    with open(ref_txt_out, "w", encoding="utf-8") as f:
        f.write(ref_item["text"])

    print(f"\nSelected Best Reference Audio for Prompting:")
    print(f"  Path    : {ref_wav_out}")
    print(f"  Text    : {ref_item['text']}")
    print(f"  Duration: {ref_item['duration']:.2f}s")

if __name__ == "__main__":
    main()
