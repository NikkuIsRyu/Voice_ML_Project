#!/usr/bin/env python3
"""
Model Fine-Tuning Script for GPT-SoVITS v2.

Fine-tunes:
  1. SoVITS Acoustic Model (s2_train.py): 8 epochs, batch_size=2, fp16
  2. GPT Semantic Model (s1_train.py): 15 epochs, batch_size=2, fp16
"""

import io
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPT_SOVITS_DIR = PROJECT_ROOT / "GPT-SoVITS"
EXP_NAME = "speaker_A4"
EXP_DIR = GPT_SOVITS_DIR / "logs" / EXP_NAME
TEMP_DIR = GPT_SOVITS_DIR / "TEMP"
PYTHON_EXE = sys.executable

SOVITS_WEIGHTS_DIR = GPT_SOVITS_DIR / "SoVITS_weights_v2"
GPT_WEIGHTS_DIR = GPT_SOVITS_DIR / "GPT_weights_v2"

def train_sovits(epochs=8, batch_size=2, save_every=4):
    print(f"\n{'='*60}")
    print(f"Step 1: Training SoVITS Model ({epochs} epochs, batch_size={batch_size})")
    print(f"{'='*60}")

    (EXP_DIR / "logs_s2_v2").mkdir(parents=True, exist_ok=True)
    SOVITS_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    config_template = GPT_SOVITS_DIR / "GPT_SoVITS" / "configs" / "s2.json"
    with open(config_template, "r", encoding="utf-8") as f:
        s2_data = json.load(f)

    pretrained_g = GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s2G2333k.pth"
    pretrained_d = GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s2D2333k.pth"

    s2_data["train"]["batch_size"] = batch_size
    s2_data["train"]["epochs"] = epochs
    s2_data["train"]["text_low_lr_rate"] = 0.4
    s2_data["train"]["pretrained_s2G"] = str(pretrained_g).replace("\\", "/")
    s2_data["train"]["pretrained_s2D"] = str(pretrained_d).replace("\\", "/")
    s2_data["train"]["if_save_latest"] = True
    s2_data["train"]["if_save_every_weights"] = True
    s2_data["train"]["save_every_epoch"] = save_every
    s2_data["train"]["gpu_numbers"] = "0"
    s2_data["train"]["grad_ckpt"] = False
    s2_data["train"]["lora_rank"] = "32"
    s2_data["model"]["version"] = "v2"
    s2_data["data"]["exp_dir"] = str(EXP_DIR).replace("\\", "/")
    s2_data["s2_ckpt_dir"] = str(EXP_DIR).replace("\\", "/")
    s2_data["save_weight_dir"] = str(SOVITS_WEIGHTS_DIR).replace("\\", "/")
    s2_data["name"] = EXP_NAME
    s2_data["version"] = "v2"

    tmp_config = TEMP_DIR / "tmp_s2.json"
    with open(tmp_config, "w", encoding="utf-8") as f:
        json.dump(s2_data, f, indent=2)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [PYTHON_EXE, "-s", "GPT_SoVITS/s2_train.py", "--config", str(tmp_config)]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(GPT_SOVITS_DIR), env=env)
    if proc.returncode != 0:
        print(f"Error: SoVITS training failed with exit code {proc.returncode}!")
        sys.exit(proc.returncode)
    print("SoVITS training completed successfully!")

def train_gpt(epochs=15, batch_size=2, save_every=5):
    print(f"\n{'='*60}")
    print(f"Step 2: Training GPT Model ({epochs} epochs, batch_size={batch_size})")
    print(f"{'='*60}")

    (EXP_DIR / "logs_s1").mkdir(parents=True, exist_ok=True)
    GPT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    config_template = GPT_SOVITS_DIR / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml"
    with open(config_template, "r", encoding="utf-8") as f:
        s1_data = yaml.load(f, Loader=yaml.FullLoader)

    pretrained_s1 = GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"

    s1_data["train"]["batch_size"] = batch_size
    s1_data["train"]["epochs"] = epochs
    s1_data["train"]["save_every_n_epoch"] = save_every
    s1_data["train"]["if_save_every_weights"] = True
    s1_data["train"]["if_save_latest"] = True
    s1_data["train"]["if_dpo"] = False
    s1_data["train"]["half_weights_save_dir"] = str(GPT_WEIGHTS_DIR).replace("\\", "/")
    s1_data["train"]["exp_name"] = EXP_NAME
    s1_data["pretrained_s1"] = str(pretrained_s1).replace("\\", "/")
    s1_data["train_semantic_path"] = str(EXP_DIR / "6-name2semantic.tsv").replace("\\", "/")
    s1_data["train_phoneme_path"] = str(EXP_DIR / "2-name2text.txt").replace("\\", "/")
    s1_data["output_dir"] = str(EXP_DIR / "logs_s1_v2").replace("\\", "/")
    s1_data["data"]["num_workers"] = 0

    tmp_config = TEMP_DIR / "tmp_s1.yaml"
    with open(tmp_config, "w", encoding="utf-8") as f:
        yaml.dump(s1_data, f)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["_CUDA_VISIBLE_DEVICES"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["hz"] = "25hz"

    cmd = [PYTHON_EXE, "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(tmp_config)]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(GPT_SOVITS_DIR), env=env)
    if proc.returncode != 0:
        print(f"Error: GPT training failed with exit code {proc.returncode}!")
        sys.exit(proc.returncode)
    print("GPT training completed successfully!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train GPT-SoVITS models")
    parser.add_argument("--sovits-only", action="store_true", help="Only train SoVITS")
    parser.add_argument("--gpt-only", action="store_true", help="Only train GPT")
    args = parser.parse_args()

    if not args.gpt_only:
        train_sovits(epochs=8, batch_size=2, save_every=4)

    if not args.sovits_only:
        train_gpt(epochs=15, batch_size=2, save_every=5)

    print(f"\n{'='*60}")
    print("Fine-Tuning Results Summary:")
    print(f"{'='*60}")
    sovits_files = list(SOVITS_WEIGHTS_DIR.glob("*.pth"))
    gpt_files = list(GPT_WEIGHTS_DIR.glob("*.ckpt"))
    print(f"SoVITS Weights ({len(sovits_files)} found in {SOVITS_WEIGHTS_DIR.name}):")
    for f in sovits_files:
        print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"GPT Weights ({len(gpt_files)} found in {GPT_WEIGHTS_DIR.name}):")
    for f in gpt_files:
        print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    main()
