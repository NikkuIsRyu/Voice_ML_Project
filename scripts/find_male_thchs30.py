import sys
from pathlib import Path
import soundfile as sf
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tutor" / "backend"))
from audio_utils import extract_f0_contour

p = (PROJECT_ROOT.parent / "Voice_models" / "datasets" / "thchs30" / "data_thchs30" / "data").resolve()
all_wavs = list(p.glob("*.wav"))
all_spks = sorted(list(set(w.name.split("_")[0] for w in all_wavs)))

print(f"Scanning {len(all_spks)} speakers in THCHS30...")
male_speakers = []
for spk in all_spks:
    spk_wavs = list(p.glob(f"{spk}_*.wav"))
    sample_wavs = spk_wavs[:6]
    f0s = []
    durations = []
    for w in sample_wavs:
        try:
            d, sr = sf.read(str(w))
            durations.append(len(d) / sr)
            info = extract_f0_contour(d, sr=sr)
            if info["has_voicing"] and info["mean_f0"] > 60:
                f0s.append(info["mean_f0"])
        except Exception:
            pass
    if f0s:
        mf0 = float(np.mean(f0s))
        if mf0 < 195.0: # clear male pitch range
            # Calculate total duration for this speaker
            total_dur = sum(durations) / len(durations) * len(spk_wavs)
            male_speakers.append((spk, mf0, len(spk_wavs), total_dur))

male_speakers.sort(key=lambda x: x[1]) # sort by pitch (deepest first)
print(f"\nFound {len(male_speakers)} male speakers:")
for spk, pitch, count, dur in male_speakers:
    print(f"  Speaker {spk:5s} | Pitch: {pitch:5.1f} Hz | Clips: {count:4d} | Est. Duration: {dur/60:4.1f} mins")
