import os
import sys
import re
import glob
from pathlib import Path
import argparse
import soundfile as sf
import numpy as np

def clean_prosody_text(raw_text):
    """
    Convert DataBaker CSMSC prosodic tags to standard Chinese punctuation:
    #1: minor word boundary (remove)
    #2: minor phrase pause (remove or comma depending on rhythm)
    #3: major phrase boundary (comma '，')
    #4: sentence boundary (period '。')
    """
    t = raw_text.strip()
    # Replace #3 with comma
    t = t.replace("#3", "，")
    # Replace #4 with period
    t = t.replace("#4", "。")
    # Replace #2 with comma if surrounded by characters, or remove
    # In Putonghua, #2 is a light pause; replacing with comma keeps natural rhythm
    t = t.replace("#2", "，")
    # Remove #1
    t = t.replace("#1", "")
    
    # Clean up double punctuation
    t = re.sub(r"[，,]+", "，", t)
    t = re.sub(r"[。.]+", "。", t)
    t = re.sub(r"[，,][。.]", "。", t)
    t = re.sub(r"^[，。？！、\s]+", "", t)
    t = t.strip()
    if t and not any(t.endswith(p) for p in "。？！…"):
        t += "。"
    return t

def normalize_loudness(audio_data, sr, target_lufs=-23.0):
    """
    RMS-based loudness normalization to approximate -23 LUFS with peak limiting.
    """
    rms = np.sqrt(np.mean(audio_data**2))
    if rms < 1e-6:
        return audio_data
    # -23 LUFS is approximately RMS of 0.0707 (-23 dBFS)
    target_rms = 10.0 ** (target_lufs / 20.0)
    gain = target_rms / rms
    norm = audio_data * gain
    # Peak limit at -1 dBFS (0.891)
    peak = np.max(np.abs(norm))
    if peak > 0.891:
        norm = norm * (0.891 / peak)
    return norm

def main():
    parser = argparse.ArgumentParser(description="Prepare Biaobei CSMSC dataset for GPT-SoVITS")
    parser.add_argument("--source_dir", default="../Voice_models/datasets/BZNSYP",
                        help="Path to unpacked BZNSYP folder")
    parser.add_argument("--output_dir", default="data/train_v2/csmsc",
                        help="Output directory for formatted dataset")
    parser.add_argument("--max_items", type=int, default=1000,
                        help="Number of sentences to prepare (default: 1000, approx. 75 min)")
    args = parser.parse_args()

    src = Path(args.source_dir).resolve()
    out = Path(args.output_dir).resolve()
    wav_out = out / "wavs"
    wav_out.mkdir(parents=True, exist_ok=True)

    # Locate ProsodyLabeling file
    label_files = list(src.glob("**/000001-010000.txt"))
    if not label_files:
        raise FileNotFoundError(f"Could not find 000001-010000.txt in {src}")
    label_path = label_files[0]
    print(f"Found labeling file: {label_path}")

    # Locate Wave directory
    wave_dirs = list(src.glob("**/Wave"))
    if not wave_dirs:
        # Check if wav files are directly in src or subfolder
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

    # In CSMSC, every odd line (0, 2, 4...) is id + text, even line is pinyin
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

    print(f"Parsed {len(items)} items from transcript.")

    # Filter items: take first max_items that have valid wav files
    selected = []
    total_duration = 0.0

    print(f"\nProcessing and normalizing audio (target: {args.max_items} clips)...")
    for item in items:
        if len(selected) >= args.max_items:
            break

        wav_name = f"{item['id']}.wav"
        src_wav = wave_dir / wav_name
        if not src_wav.exists():
            continue

        try:
            data, sr = sf.read(str(src_wav))
            # Convert stereo to mono if needed
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            dur = len(data) / sr
            # Keep clips between 1.5s and 12s (ideal for GPT-SoVITS)
            if dur < 1.5 or dur > 12.0:
                continue

            # Normalize loudness
            norm_data = normalize_loudness(data, sr, target_lufs=-23.0)

            # Target output wav
            tgt_wav = wav_out / wav_name
            sf.write(str(tgt_wav), norm_data, sr)

            selected.append({
                **item,
                "wav_path": str(tgt_wav).replace("\\", "/"),
                "duration": dur
            })
            total_duration += dur

            if len(selected) % 100 == 0:
                print(f"[{len(selected)}/{args.max_items}] Processed {total_duration/60:.1f} mins of audio...")
        except Exception as e:
            print(f"Error processing {src_wav}: {e}")

    print(f"\nCompleted selection: {len(selected)} items, {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours).")

    # Write GPT-SoVITS training list: 2-name2text.txt
    # Format: {wav_path}|{speaker_name}|{language}|{text}
    name2text_path = out / "2-name2text.txt"
    with open(name2text_path, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(f"{item['wav_path']}|csmsc|zh|{item['text']}\n")

    print(f"Wrote metadata: {name2text_path}")

    # Select best reference clips (between 4.0s and 6.5s, clean tone)
    ref_candidates = [
        item for item in selected
        if 4.0 <= item["duration"] <= 6.5 and len(item["text"]) >= 10 and len(item["text"]) <= 22
    ]
    ref_item = ref_candidates[0] if ref_candidates else selected[0]

    # Save reference audio and prompt text
    ref_wav_out = out / "csmsc_ref.wav"
    ref_txt_out = out / "csmsc_ref.txt"

    data, sr = sf.read(ref_item["wav_path"])
    sf.write(str(ref_wav_out), data, sr)
    with open(ref_txt_out, "w", encoding="utf-8") as f:
        f.write(ref_item["text"])

    print(f"\nSaved Reference Audio for Zero-Shot & Prompting:")
    print(f"- Wav: {ref_wav_out}")
    print(f"- Text: '{ref_item['text']}' (Duration: {ref_item['duration']:.2f}s)")
    print("\nDataset preparation script finished successfully!")

if __name__ == "__main__":
    main()
