#!/usr/bin/env python3
"""
Feature Extraction Pipeline for Biaobei CSMSC (v2ProPlus Fine-Tuning).

Extracts:
  1. Text cleaner + Chinese RoBERTa embeddings (1-get-text.py)
  2. HuBERT SSL features + 32kHz audio resample (2-get-hubert-wav32k.py)
  3. Semantic tokens extraction using v2ProPlus base model (3-get-semantic.py)
"""

import io
import os
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPT_SOVITS_DIR = PROJECT_ROOT / "GPT-SoVITS"
DATA_DIR = PROJECT_ROOT / "data" / "train_v2" / "csmsc"

EXP_NAME = "csmsc_v2"
OPT_DIR = GPT_SOVITS_DIR / "logs" / EXP_NAME
INP_TEXT = DATA_DIR / "2-name2text.txt"
INP_WAV_DIR = DATA_DIR / "wavs"

BERT_PRETRAINED = GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"
CNHUBERT_BASE = GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"
PRETRAINED_S2G = GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Gv2ProPlus.pth"
S2_CONFIG = GPT_SOVITS_DIR / "GPT_SoVITS" / "configs" / "s2v2ProPlus.json"

PYTHON_EXE = sys.executable

def run_step(step_name, script_rel, env_vars):
    print(f"\n{'='*60}")
    print(f"Starting {step_name}: {script_rel}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env.update(env_vars)
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [PYTHON_EXE, "-s", str(GPT_SOVITS_DIR / script_rel)]
    proc = subprocess.run(cmd, cwd=str(GPT_SOVITS_DIR), env=env)
    if proc.returncode != 0:
        print(f"Error: {step_name} failed with return code {proc.returncode}!")
        sys.exit(proc.returncode)
    print(f"{step_name} finished successfully.")

def main():
    OPT_DIR.mkdir(parents=True, exist_ok=True)

    base_env = {
        "inp_text": str(INP_TEXT),
        "inp_wav_dir": str(INP_WAV_DIR),
        "exp_name": EXP_NAME,
        "opt_dir": str(OPT_DIR),
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "is_half": "True",
        "version": "v2",
    }

    # Step 1: 1-get-text.py
    bert_dir = OPT_DIR / "3-bert"
    if not (bert_dir.exists() and len(list(bert_dir.glob("*.pt"))) >= 1000):
        step1_env = dict(base_env)
        step1_env["bert_pretrained_dir"] = str(BERT_PRETRAINED)
        run_step("Step 1a (Text + BERT)", "GPT_SoVITS/prepare_datasets/1-get-text.py", step1_env)

        # Merge step 1 outputs
        txt_part = OPT_DIR / "2-name2text-0.txt"
        txt_all = OPT_DIR / "2-name2text.txt"
        if txt_part.exists():
            content = txt_part.read_text(encoding="utf-8")
            txt_all.write_text(content, encoding="utf-8")
            txt_part.unlink()
            print(f"Merged text annotations into {txt_all}")
    else:
        print("Step 1a (Text + BERT) already complete (1,000 files found), skipping.")

    # Step 2: 2-get-hubert-wav32k.py
    hubert_dir = OPT_DIR / "4-cnhubert"
    if not (hubert_dir.exists() and len(list(hubert_dir.glob("*.pt"))) >= 1000):
        step2_env = dict(base_env)
        step2_env["cnhubert_base_dir"] = str(CNHUBERT_BASE)
        step2_env["sv_path"] = str(GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt")
        run_step("Step 1b (HuBERT + 32k Wav)", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py", step2_env)
    else:
        print("Step 1b (HuBERT + 32k Wav) already complete (1,000 files found), skipping.")

    # Step 3: 3-get-semantic.py
    sem_all = OPT_DIR / "6-name2semantic.tsv"
    if not (sem_all.exists() and sem_all.stat().st_size > 100000):
        step3_env = dict(base_env)
        step3_env["pretrained_s2G"] = str(PRETRAINED_S2G)
        step3_env["s2config_path"] = str(S2_CONFIG)
        run_step("Step 1c (Semantic Tokens via v2ProPlus)", "GPT_SoVITS/prepare_datasets/3-get-semantic.py", step3_env)

        # Merge step 3 outputs
        sem_part = OPT_DIR / "6-name2semantic-0.tsv"
        if sem_part.exists():
            content = sem_part.read_text(encoding="utf-8")
            sem_all.write_text(content, encoding="utf-8")
            sem_part.unlink()
            print(f"Merged semantic tokens into {sem_all}")
    else:
        print("Step 1c (Semantic Tokens) already complete, skipping.")

    # Step 4: 2-get-sv.py (Speaker Verification features for v2Pro/v2ProPlus)
    sv_dir = OPT_DIR / "7-sv_cn"
    if not (sv_dir.exists() and len(list(sv_dir.glob("*.pt"))) >= 1000):
        step4_env = dict(base_env)
        step4_env["sv_path"] = str(GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt")
        run_step("Step 1d (Speaker Verification features for v2ProPlus)", "GPT_SoVITS/prepare_datasets/2-get-sv.py", step4_env)
    else:
        print("Step 1d (Speaker Verification) already complete (1,000 files found), skipping.")

    # Verify all expected outputs
    print(f"\n{'='*60}")
    print("Verifying Dataset Formatting Outputs:")
    print(f"{'='*60}")
    for item in ["2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k", "6-name2semantic.tsv", "7-sv_cn"]:
        p = OPT_DIR / item
        exists = p.exists()
        count = len(list(p.iterdir())) if p.is_dir() else (p.stat().st_size if exists else 0)
        unit = "files" if p.is_dir() else "bytes"
        print(f"  {item:20}: {'EXISTS' if exists else 'MISSING'} ({count} {unit})")

if __name__ == "__main__":
    main()
