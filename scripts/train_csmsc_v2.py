#!/usr/bin/env python3
"""
Full Fine-Tuning Pipeline for Biaobei CSMSC on v2ProPlus.

Executes:
  1. SoVITS Training (s2_train.py) with v2ProPlus generator & discriminator.
  2. GPT Training (s1_train.py) with longer-v2 base checkpoint.
  3. Copies checkpoints to persistent storage and updates tts_infer.yaml.
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPT_SOVITS_DIR = PROJECT_ROOT / "GPT-SoVITS"
VOICE_MODELS_DIR = PROJECT_ROOT.parent / "Voice_models"

EXP_NAME = "csmsc_v2"
VERSION = "v2ProPlus"
EXP_DIR = GPT_SOVITS_DIR / "logs" / EXP_NAME

PRETRAINED_S2G = GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Gv2ProPlus.pth"
PRETRAINED_S2D = GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Dv2ProPlus.pth"
PRETRAINED_S1 = GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"

SOVITS_WEIGHT_DIR = GPT_SOVITS_DIR / "SoVITS_weights_v2ProPlus"
GPT_WEIGHT_DIR = GPT_SOVITS_DIR / "GPT_weights_v2ProPlus"

PYTHON_EXE = sys.executable

def train_sovits(epochs=10, batch_size=2, save_every=2):
    print("\n" + "="*70)
    print("PHASE 12 - STEP 1: SoVITS Fine-Tuning (v2ProPlus)")
    print(f"Dataset: 1,000 samples | Epochs: {epochs} | Batch size: {batch_size}")
    print("="*70)

    SOVITS_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    logs_s2 = EXP_DIR / f"logs_s2_{VERSION}"
    logs_s2.mkdir(parents=True, exist_ok=True)

    base_config_path = GPT_SOVITS_DIR / "GPT_SoVITS" / "configs" / "s2v2ProPlus.json"
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Configure training
    config["train"]["epochs"] = epochs
    config["train"]["batch_size"] = batch_size
    config["train"]["save_every_epoch"] = save_every
    config["train"]["fp16_run"] = True
    config["train"]["learning_rate"] = 0.0001
    config["train"]["text_low_lr_rate"] = 0.4
    config["train"]["pretrained_s2G"] = str(PRETRAINED_S2G)
    config["train"]["pretrained_s2D"] = str(PRETRAINED_S2D)
    config["train"]["if_save_latest"] = True
    config["train"]["if_save_every_weights"] = True
    config["train"]["gpu_numbers"] = "0"
    config["train"]["grad_ckpt"] = False
    config["train"]["lora_rank"] = None

    # Model and paths
    config["model"]["version"] = VERSION
    config["data"]["exp_dir"] = str(EXP_DIR)
    config["s2_ckpt_dir"] = str(EXP_DIR)
    config["save_weight_dir"] = str(SOVITS_WEIGHT_DIR)
    config["name"] = EXP_NAME
    config["version"] = VERSION

    train_config_file = EXP_DIR / "train_s2.json"
    with open(train_config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    ffmpeg_dir = str(GPT_SOVITS_DIR)
    conda_bin = str(Path(PYTHON_EXE).parent / "Library" / "bin")
    env["PATH"] = f"{ffmpeg_dir};{conda_bin};" + env.get("PATH", "")

    cmd = [PYTHON_EXE, "-s", "GPT_SoVITS/s2_train.py", "--config", str(train_config_file)]
    print(f"Executing: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(GPT_SOVITS_DIR), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"SoVITS training failed with exit code {proc.returncode}")
    print("SoVITS training completed successfully.")

def train_gpt(epochs=15, batch_size=4, save_every=5):
    print("\n" + "="*70)
    print("PHASE 12 - STEP 2: GPT Semantic Token Fine-Tuning")
    print(f"Dataset: 1,000 samples | Epochs: {epochs} | Batch size: {batch_size}")
    print("="*70)

    GPT_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    logs_s1 = EXP_DIR / f"logs_s1_{VERSION}"
    logs_s1.mkdir(parents=True, exist_ok=True)

    base_yaml_path = GPT_SOVITS_DIR / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml"
    with open(base_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Configure training
    config["train"]["epochs"] = epochs
    config["train"]["batch_size"] = batch_size
    config["train"]["save_every_n_epoch"] = save_every
    config["train"]["precision"] = "16-mixed"
    config["train"]["if_save_latest"] = True
    config["train"]["if_save_every_weights"] = True
    config["train"]["if_dpo"] = False
    config["train"]["half_weights_save_dir"] = str(GPT_WEIGHT_DIR)
    config["train"]["exp_name"] = EXP_NAME

    config["pretrained_s1"] = str(PRETRAINED_S1)
    config["train_semantic_path"] = str(EXP_DIR / "6-name2semantic.tsv")
    config["train_phoneme_path"] = str(EXP_DIR / "2-name2text.txt")
    config["output_dir"] = str(logs_s1)

    # Windows DataLoader stability: set num_workers to 0
    config["data"]["num_workers"] = 0

    train_config_file = EXP_DIR / "train_s1.yaml"
    with open(train_config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["_CUDA_VISIBLE_DEVICES"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["hz"] = "25hz"
    ffmpeg_dir = str(GPT_SOVITS_DIR)
    conda_bin = str(Path(PYTHON_EXE).parent / "Library" / "bin")
    env["PATH"] = f"{ffmpeg_dir};{conda_bin};" + env.get("PATH", "")

    cmd = [PYTHON_EXE, "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(train_config_file)]
    print(f"Executing: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(GPT_SOVITS_DIR), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"GPT training failed with exit code {proc.returncode}")
    print("GPT training completed successfully.")

def update_infer_config():
    print("\n" + "="*70)
    print("Updating tts_infer.yaml with newly trained v2ProPlus weights")
    print("="*70)

    # Find latest sovits weight
    sovits_ckpts = sorted(SOVITS_WEIGHT_DIR.glob(f"{EXP_NAME}*.pth"), key=os.path.getmtime)
    if not sovits_ckpts:
        print(f"Warning: No weights found in {SOVITS_WEIGHT_DIR}")
        return
    best_sovits = sovits_ckpts[-1]

    # Find latest gpt weight
    gpt_ckpts = sorted(GPT_WEIGHT_DIR.glob(f"{EXP_NAME}*.ckpt"), key=os.path.getmtime)
    if not gpt_ckpts:
        print(f"Warning: No weights found in {GPT_WEIGHT_DIR}")
        return
    best_gpt = gpt_ckpts[-1]

    print(f"Selected SoVITS weight: {best_sovits.name}")
    print(f"Selected GPT weight:    {best_gpt.name}")

    # Backup to Voice_models
    backup_v2 = VOICE_MODELS_DIR / "v2"
    backup_v2.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_sovits, backup_v2 / best_sovits.name)
    shutil.copy2(best_gpt, backup_v2 / best_gpt.name)
    print(f"Backed up checkpoints to: {backup_v2}")

    # Update tts_infer.yaml custom section
    infer_yaml_path = GPT_SOVITS_DIR / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    with open(infer_yaml_path, "r", encoding="utf-8") as f:
        infer_cfg = yaml.safe_load(f)

    infer_cfg["custom"]["vits_weights_path"] = f"SoVITS_weights_v2ProPlus/{best_sovits.name}"
    infer_cfg["custom"]["t2s_weights_path"] = f"GPT_weights_v2ProPlus/{best_gpt.name}"
    infer_cfg["custom"]["version"] = VERSION

    with open(infer_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(infer_cfg, f, default_flow_style=False)
    print(f"Updated {infer_yaml_path} to use {best_sovits.name} and {best_gpt.name}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-SoVITS v2ProPlus on Biaobei CSMSC dataset.")
    parser.add_argument("--sovits_epochs", type=int, default=10, help="Epochs for SoVITS")
    parser.add_argument("--gpt_epochs", type=int, default=15, help="Epochs for GPT")
    parser.add_argument("--sovits_batch_size", type=int, default=2, help="Batch size for SoVITS")
    parser.add_argument("--gpt_batch_size", type=int, default=4, help="Batch size for GPT")
    parser.add_argument("--skip_sovits", action="store_true", help="Skip SoVITS training")
    parser.add_argument("--skip_gpt", action="store_true", help="Skip GPT training")
    args = parser.parse_args()

    if not args.skip_sovits:
        train_sovits(epochs=args.sovits_epochs, batch_size=args.sovits_batch_size, save_every=2)
    if not args.skip_gpt:
        train_gpt(epochs=args.gpt_epochs, batch_size=args.gpt_batch_size, save_every=5)

    update_infer_config()

    print("\n" + "="*70)
    print("ALL TRAINING STAGES COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
