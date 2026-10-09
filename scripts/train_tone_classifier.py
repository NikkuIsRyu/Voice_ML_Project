#!/usr/bin/env python3
"""
Train Tone Classifier on CSMSC (BZNSYP) and THCHS-30 Datasets.
Extracts 15-dim acoustic pitch features per syllable:
- 10-point normalized semitone contour (relative to utterance median pitch)
- Duration (seconds)
- Contour slope
- Mean semitone, min semitone, max semitone, pitch range

Acceptance Criteria:
- Overall accuracy >= 85%
- Save model to Voice_models/tone_clf/tone_model.pkl
- Save confusion matrix and metrics to eval/tone_clf/
"""

import os
import sys
import glob
import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSMSC_DIR = PROJECT_ROOT.parent / "Voice_models" / "datasets" / "BZNSYP"
OUTPUT_MODEL_DIR = PROJECT_ROOT.parent / "Voice_models" / "tone_clf"
EVAL_DIR = PROJECT_ROOT / "eval" / "tone_clf"

OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
HOP_LENGTH = 256
FRAME_DUR = HOP_LENGTH / SR  # 0.016s

INITIAL_SET = {
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"
}

def parse_interval_file(filepath: Path) -> List[Tuple[float, float, str]]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines()]
    if len(lines) < 13:
        return []
    try:
        n_intervals = int(lines[11])
    except ValueError:
        return []

    intervals = []
    idx = 12
    for _ in range(n_intervals):
        if idx + 2 >= len(lines):
            break
        try:
            t0 = float(lines[idx])
            t1 = float(lines[idx + 1])
            lbl = lines[idx + 2].strip('\" \t')
            intervals.append((t0, t1, lbl))
        except ValueError:
            pass
        idx += 3
    return intervals

def group_syllables(intervals: List[Tuple[float, float, str]]) -> List[Dict]:
    syllables = []
    prev_initial = None

    for t0, t1, lbl in intervals:
        if lbl in ("sil", "sp", ""):
            prev_initial = None
            continue

        if lbl[-1] in "12345":
            tone = int(lbl[-1])
            start_t = prev_initial[0] if prev_initial else t0
            end_t = t1
            syllables.append({
                "start": start_t,
                "end": end_t,
                "duration": end_t - start_t,
                "tone": tone,
                "phone": lbl
            })
            prev_initial = None
        elif lbl in INITIAL_SET:
            prev_initial = (t0, t1, lbl)
        else:
            prev_initial = None

    return syllables

def extract_utterance_features(wav_path: Path, intervals: List[Tuple[float, float, str]]) -> List[Tuple[np.ndarray, int]]:
    try:
        y, file_sr = sf.read(str(wav_path))
    except Exception:
        return []

    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)

    if file_sr != SR:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=SR)

    # Fast YIN pitch
    f0 = librosa.yin(y, fmin=65, fmax=1046, sr=SR, frame_length=1024, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP_LENGTH)[0]

    min_l = min(len(f0), len(rms))
    f0 = f0[:min_l]
    rms = rms[:min_l]

    valid_mask = (rms > 0.012) & (f0 > 70) & (f0 < 900)
    if np.sum(valid_mask) < 5:
        return []

    median_f0 = float(np.median(f0[valid_mask]))
    syllables = group_syllables(intervals)
    features_list = []

    for syl in syllables:
        dur = syl["duration"]
        if dur < 0.04 or dur > 1.2:
            continue

        f_start = max(0, int(syl["start"] / FRAME_DUR))
        f_end = min(len(f0), int(syl["end"] / FRAME_DUR) + 1)
        if f_end <= f_start:
            continue

        syl_f0 = f0[f_start:f_end]
        syl_valid = valid_mask[f_start:f_end]

        voiced_indices = np.where(syl_valid)[0]
        if len(voiced_indices) < 2:
            continue

        voiced_f0 = syl_f0[voiced_indices]
        # Semitones relative to utterance median
        semitones = 12.0 * np.log2(voiced_f0 / (median_f0 + 1e-6))

        # Interpolate to exactly 10 contour points
        orig_x = np.linspace(0, 1, len(semitones))
        target_x = np.linspace(0, 1, 10)
        contour_10 = np.interp(target_x, orig_x, semitones)

        slope = float((contour_10[-1] - contour_10[0]) / (dur + 1e-5))
        mean_semi = float(np.mean(contour_10))
        min_semi = float(np.min(contour_10))
        max_semi = float(np.max(contour_10))
        delta_semi = float(max_semi - min_semi)
        curv = float(contour_10[5] - 0.5 * (contour_10[0] + contour_10[-1]))
        half1_slope = float((contour_10[4] - contour_10[0]) / (0.5 * dur + 1e-5))
        half2_slope = float((contour_10[-1] - contour_10[5]) / (0.5 * dur + 1e-5))
        min_pos = float(np.argmin(contour_10)) / 9.0
        max_pos = float(np.argmax(contour_10)) / 9.0

        feat_vector = np.concatenate([
            contour_10,
            np.array([dur, slope, curv, half1_slope, half2_slope, mean_semi, min_semi, max_semi, delta_semi, min_pos, max_pos], dtype=np.float32)
        ])

        features_list.append((feat_vector, syl["tone"]))

    return features_list

def main():
    print("=== Training Tone Classifier (CSMSC BZNSYP + Evaluation) ===")
    interval_files = sorted(glob.glob(str(CSMSC_DIR / "PhoneLabeling" / "*.interval")))
    print(f"Found {len(interval_files)} CSMSC interval files.")

    cache_path = PROJECT_ROOT / "data" / "csmsc_tone_features.npz"
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}...")
        npz = np.load(cache_path)
        X = npz["X"]
        y = npz["y"]
        processed_count = 1500
    else:
        # Process first 1,500 utterances to extract ~20,000 syllables
        max_utts = 1500
        X_list = []
        y_list = []

        t0 = time.time()
        processed_count = 0

        for i, int_path in enumerate(interval_files[:max_utts]):
            p = Path(int_path)
            wav_path = CSMSC_DIR / "Wave" / f"{p.stem}.wav"
            if not wav_path.exists():
                continue

            intervals = parse_interval_file(p)
            if not intervals:
                continue

            feats = extract_utterance_features(wav_path, intervals)
            for feat_vec, tone_label in feats:
                X_list.append(feat_vec)
                y_list.append(tone_label)

            processed_count += 1
            if (i + 1) % 250 == 0:
                print(f"Processed {i + 1}/{max_utts} utterances ({len(X_list)} syllables extracted)...")

        print(f"Extraction completed in {time.time() - t0:.2f}s.")
        print(f"Total labeled syllables: {len(X_list)}")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"Cached extracted features to {cache_path}")

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique.tolist(), counts.tolist())))

    # Split 80/20 train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training enhanced MLP classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        max_iter=400,
        random_state=42,
        verbose=False
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> Test Accuracy: {acc * 100.0:.2f}% <<<")

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Save model and scaler
    model_payload = {
        "scaler": scaler,
        "model": clf,
        "classes": clf.classes_.tolist(),
        "accuracy": float(acc)
    }
    model_save_path = OUTPUT_MODEL_DIR / "tone_model.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(model_payload, f)
    print(f"Model saved to: {model_save_path}")

    # Save metrics and confusion matrix
    cm_payload = {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "classes": clf.classes_.tolist(),
        "report": report
    }
    cm_path = EVAL_DIR / "confusion_matrix.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm_payload, f, indent=2)

    # Save Markdown Report
    md_report = f"""# Tone Classifier Evaluation Report

- **Model:** Multilayer Perceptron (128, 64, ReLU)
- **Dataset:** CSMSC (BZNSYP) Phone & Tone Interval Labels ({processed_count} utterances)
- **Total Syllables:** {len(X)}
- **Test Set Accuracy:** **{acc * 100.0:.2f}%** (Target >= 85%)

## Classification Breakdown

| Tone | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
"""
    for tone_id in [1, 2, 3, 4, 5]:
        s_id = str(tone_id)
        if s_id in report:
            r = report[s_id]
            md_report += f"| Tone {tone_id} | {r['precision']*100:.1f}% | {r['recall']*100:.1f}% | {r['f1-score']*100:.1f}% | {r['support']} |\n"

    md_report += f"""
## Confusion Matrix (Rows: Actual Tone, Columns: Predicted Tone)

```
{cm}
```
"""
    with open(EVAL_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"Report written to: {EVAL_DIR / 'summary.md'}")

if __name__ == "__main__":
    main()
