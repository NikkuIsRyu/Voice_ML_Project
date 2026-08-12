#!/bin/bash
# =============================================================================
# GPT-SoVITS PC Setup Script
# =============================================================================
# Run this on your desktop PC (with RTX 4060) to set up the environment.
# This assumes you've already cloned the project repo via git.
#
# Usage:
#   chmod +x scripts/setup_pc.sh
#   ./scripts/setup_pc.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ML_MODELS_DIR="$HOME/ml_models"
GPT_SOVITS_DIR="$PROJECT_DIR/GPT-SoVITS"

echo "============================================"
echo "  GPT-SoVITS PC Setup"
echo "============================================"
echo "Project dir:  $PROJECT_DIR"
echo "Models dir:   $ML_MODELS_DIR"
echo "GPT-SoVITS:   $GPT_SOVITS_DIR"
echo ""

# --- Step 1: System dependencies ---
echo "[1/8] Installing system dependencies..."
sudo apt update && sudo apt install -y git git-lfs ffmpeg libsox-dev build-essential python3-dev gcc g++ wget curl

# --- Step 2: Check conda ---
echo "[2/8] Checking conda..."
if ! command -v conda &> /dev/null; then
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "Conda not found. Installing Miniconda..."
        mkdir -p ~/miniconda3
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm ~/miniconda3/miniconda.sh
        ~/miniconda3/bin/conda init bash
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
else
    echo "Conda found: $(conda --version)"
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"

# --- Step 3: Clone GPT-SoVITS if not present ---
echo "[3/8] Checking GPT-SoVITS..."
if [ ! -d "$GPT_SOVITS_DIR" ]; then
    echo "Cloning GPT-SoVITS..."
    git clone https://github.com/RVC-Boss/GPT-SoVITS.git "$GPT_SOVITS_DIR"
else
    echo "GPT-SoVITS already cloned."
fi

# --- Step 4: Create conda environment ---
echo "[4/8] Setting up conda environment..."
if conda env list | grep -q "GPTSoVits"; then
    echo "Conda env GPTSoVits already exists."
else
    conda create -n GPTSoVits python=3.10 -y
fi
conda activate GPTSoVits

# --- Step 5: Install PyTorch + dependencies ---
echo "[5/8] Installing PyTorch + dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

cd "$GPT_SOVITS_DIR"
pip install -r extra-req.txt --no-deps
pip install -r requirements.txt
pip install onnxruntime
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"

# --- Step 6: Create model directories + symlinks ---
echo "[6/8] Setting up model directories..."
mkdir -p "$ML_MODELS_DIR"/{pretrained,uvr5,asr,output,logs,TEMP}

# Create symlinks (remove existing dirs/links first)
for link_pair in \
    "$GPT_SOVITS_DIR/GPT_SoVITS/pretrained_models:$ML_MODELS_DIR/pretrained" \
    "$GPT_SOVITS_DIR/tools/uvr5/uvr5_weights:$ML_MODELS_DIR/uvr5" \
    "$GPT_SOVITS_DIR/tools/asr/models:$ML_MODELS_DIR/asr" \
    "$GPT_SOVITS_DIR/output:$ML_MODELS_DIR/output" \
    "$GPT_SOVITS_DIR/logs:$ML_MODELS_DIR/logs" \
    "$GPT_SOVITS_DIR/TEMP:$ML_MODELS_DIR/TEMP"; do

    link="${link_pair%%:*}"
    target="${link_pair##*:}"
    rm -rf "$link"
    mkdir -p "$(dirname "$link")"
    ln -s "$target" "$link"
    echo "  Linked: $(basename "$link") → $target"
done

# --- Step 7: Download pretrained models ---
echo "[7/8] Downloading pretrained models..."
pip install huggingface_hub

# Core models
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lj1995/GPT-SoVITS', allow_patterns=['chinese-hubert-base/*', 'chinese-roberta-wwm-ext-large/*'], local_dir='$ML_MODELS_DIR/pretrained')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lj1995/GPT-SoVITS', allow_patterns=['gsv-v2final-pretrained/*'], local_dir='$ML_MODELS_DIR/pretrained')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lj1995/GPT-SoVITS', allow_patterns=['sv/*'], local_dir='$ML_MODELS_DIR/pretrained')"

# G2PW
python -c "
from huggingface_hub import hf_hub_download
import zipfile, os
target = '$GPT_SOVITS_DIR/GPT_SoVITS/text'
if not os.path.exists(os.path.join(target, 'G2PWModel')):
    f = hf_hub_download(repo_id='XXXXRT/GPT-SoVITS-Pretrained', filename='G2PWModel.zip', local_dir='/tmp/g2pw_dl')
    with zipfile.ZipFile(f, 'r') as z:
        z.extractall(target)
    print('G2PW extracted')
else:
    print('G2PW already exists')
"

# UVR5
python -c "
from huggingface_hub import snapshot_download
import shutil, os
tmp = '$ML_MODELS_DIR/uvr5_tmp'
snapshot_download(repo_id='lj1995/VoiceConversionWebUI', allow_patterns=['uvr5_weights/*'], local_dir=tmp)
src = os.path.join(tmp, 'uvr5_weights')
if os.path.exists(src):
    for f in os.listdir(src):
        shutil.move(os.path.join(src, f), '$ML_MODELS_DIR/uvr5/')
    shutil.rmtree(tmp)
"

# ASR
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-large-v3', local_dir='$ML_MODELS_DIR/asr/faster-whisper-large-v3')"

# --- Step 8: Verify ---
echo "[8/8] Verifying installation..."
python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
    print(f'VRAM:     {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To start the WebUI:"
echo "  cd $GPT_SOVITS_DIR"
echo "  conda activate GPTSoVits"
echo "  python webui.py"
echo ""
echo "To start the API server:"
echo "  python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml"
echo ""
