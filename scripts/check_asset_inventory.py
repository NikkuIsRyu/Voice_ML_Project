import sys
from pathlib import Path

for path_str in [
    "output_v2/dialogues",
    "output_v2_slow/dialogues",
    "output_v2/vocab",
    "output_v2_slow/vocab",
    "output_v2/anki",
]:
    p = Path(path_str)
    if not p.exists():
        print(f"{path_str}: NOT FOUND")
        continue
    all_files = list(p.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    wavs = [f for f in files_only if f.suffix.lower() == ".wav"]
    size_mb = sum(f.stat().st_size for f in files_only) / (1024 * 1024)
    print(f"{path_str}: {len(files_only)} total files ({len(wavs)} wav), {size_mb:.2f} MB")
