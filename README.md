# 🎙️ Voice ML — Chinese TTS Learning Assistant

Train an AI voice to read Chinese dialogues and vocabulary aloud, helping you learn Mandarin through natural-sounding audio.

## Overview

This project uses [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) to clone a female Mandarin voice and generate audio for custom Chinese learning materials (vocab lists, dialogues from exercise books).

### How It Works
1. **Reference voice**: 60 clips of a female Mandarin speaker (A4 from THCHS-30 dataset)
2. **Fine-tune**: Train GPT-SoVITS on the reference voice (~1 hour on RTX 4060)
3. **Generate**: Type any Chinese text → get natural audio in that voice
4. **Learn**: Listen to your exercise book content read aloud with correct pronunciation

## Project Structure

```
Voice_ML_Project/
├── data/
│   ├── reference_audio/   # 60 training clips (speaker A4, female, 10.3 min)
│   ├── vocab/             # Your exercise book vocabulary (one .txt per lesson)
│   └── dialogues/         # Your exercise book dialogues (one .txt per lesson)
├── scripts/
│   ├── generate_audio.py  # Batch TTS generation script
│   └── setup_pc.sh        # One-click PC setup (for desktop with GPU)
├── output/
│   ├── vocab/             # Generated vocab audio
│   └── dialogues/         # Generated dialogue audio
├── docs/
│   └── LEARNING_GUIDE.md  # What to learn + homework
├── GPT-SoVITS/            # Cloned repo (in .gitignore)
└── README.md
```

## Quick Start

### 1. Set Up (PC with NVIDIA GPU)
```bash
git clone <your-repo-url>
cd Voice_ML_Project
chmod +x scripts/setup_pc.sh
./scripts/setup_pc.sh
```

### 2. Train the Voice (WebUI)
```bash
cd GPT-SoVITS
conda activate GPTSoVits
python webui.py
# Use the WebUI to train on data/reference_audio/
```

### 3. Generate Audio
```bash
# Start the API server
cd GPT-SoVITS
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml

# Generate vocab audio
cd ..
python scripts/generate_audio.py -i data/vocab/lesson01.txt -o output/vocab/lesson01

# Generate dialogue audio
python scripts/generate_audio.py -i data/dialogues/ -o output/dialogues/
```

### 4. Adding Your Content
See the example files in `data/vocab/` and `data/dialogues/` for the format.
See `docs/LEARNING_GUIDE.md` for a full guide.

## Requirements

- **Training**: NVIDIA GPU with ≥6 GB VRAM (RTX 4060 tested)
- **Inference**: GPU recommended, CPU works but slower
- **OS**: Ubuntu Linux
- **Storage**: ~20 GB free (models + training data)

## Tech Stack

- **GPT-SoVITS** v2 — Voice cloning & TTS
- **PyTorch** 2.5.1 + CUDA 12.1
- **Python** 3.10 (Miniconda)
- **Training data**: THCHS-30 speaker A4 (CC BY-SA 4.0)

## Model Storage

Large model files are stored in `~/ml_models/` (not synced via Seafile/Git):
- `~/ml_models/pretrained/` — HuBERT, RoBERTa, V2 base models
- `~/ml_models/uvr5/` — Vocal separation models
- `~/ml_models/asr/` — Whisper ASR model
- `~/ml_models/output/` — Training checkpoints
