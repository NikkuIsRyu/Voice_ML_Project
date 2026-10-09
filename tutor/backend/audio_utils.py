import io
import os
import shutil
import subprocess
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional, Dict, Any

TARGET_SR = 16000

def get_ffmpeg_path() -> Optional[str]:
    """Locate ffmpeg binary in GPT-SoVITS directory or system PATH."""
    cand = Path(__file__).resolve().parent.parent.parent / "GPT-SoVITS" / "ffmpeg.exe"
    if cand.exists():
        return str(cand)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    return None

def load_and_resample(
    file_bytes_or_path,
    target_sr: int = TARGET_SR,
    min_duration: float = 0.3,
    min_rms: float = 0.003
) -> Tuple[np.ndarray, int]:
    """
    Load audio from file path or bytes, convert to mono, and resample to target_sr.
    Decodes WebM, OGG, MP4, WAV using ffmpeg subprocess with soundfile fallback.
    Validates minimum duration (default >= 0.3s) and speech presence (RMS >= 0.003).
    """
    ffmpeg_bin = get_ffmpeg_path()
    data = None
    sr = target_sr

    # Attempt decoding with ffmpeg
    if ffmpeg_bin:
        try:
            if isinstance(file_bytes_or_path, (bytes, bytearray)):
                cmd = [
                    ffmpeg_bin, "-nostdin", "-v", "error",
                    "-i", "pipe:0",
                    "-ac", "1",
                    "-ar", str(target_sr),
                    "-f", "f32le",
                    "pipe:1"
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout_data, stderr_data = proc.communicate(input=bytes(file_bytes_or_path))
            else:
                cmd = [
                    ffmpeg_bin, "-nostdin", "-v", "error",
                    "-i", str(file_bytes_or_path),
                    "-ac", "1",
                    "-ar", str(target_sr),
                    "-f", "f32le",
                    "pipe:1"
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout_data, stderr_data = proc.communicate()

            if proc.returncode == 0 and len(stdout_data) > 0:
                data = np.frombuffer(stdout_data, dtype=np.float32)
                sr = target_sr
        except Exception:
            data = None

    # Fallback to soundfile if ffmpeg didn't decode
    if data is None:
        try:
            if isinstance(file_bytes_or_path, (bytes, bytearray)):
                bio = io.BytesIO(file_bytes_or_path)
                data, sr = sf.read(bio)
            else:
                data, sr = sf.read(str(file_bytes_or_path))

            if data.ndim > 1:
                data = np.mean(data, axis=1)

            data = data.astype(np.float32)

            if sr != target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
        except Exception as e:
            raise ValueError(f"Failed to decode audio: {e}")

    if data is None or len(data) == 0:
        raise ValueError("Audio data is empty")

    duration = len(data) / sr
    if duration < min_duration:
        raise ValueError(f"Recording too short ({duration:.2f}s). Please speak for at least {min_duration}s.")

    rms = float(np.sqrt(np.mean(data ** 2)))
    if rms < min_rms:
        raise ValueError(f"No speech detected (audio is silent, RMS={rms:.5f}). Please check your microphone.")

    # Normalize amplitude
    max_val = float(np.max(np.abs(data)))
    if max_val > 1e-4:
        data = data / max_val * 0.95

    return data, sr

def extract_f0_contour(audio: np.ndarray, sr: int = TARGET_SR, n_points: int = 100) -> Dict[str, Any]:
    """
    Extract F0 pitch contour using fast librosa.yin algorithm with RMS energy-gated voicing.
    Converts to semitones, normalizes trajectory, and interpolates to n_points for visual overlay.
    """
    fmin = librosa.note_to_hz("C2")  # ~65 Hz
    fmax = librosa.note_to_hz("C6")  # ~1046 Hz

    f0 = librosa.yin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=1024,
        hop_length=256
    )

    # Valid voiced frames: energy gate + bounds check
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    # Ensure length match
    min_len = min(len(f0), len(rms))
    f0 = f0[:min_len]
    rms = rms[:min_len]

    valid_mask = (rms > 0.012) & (f0 > fmin * 1.02) & (f0 < fmax * 0.98)
    if not np.any(valid_mask):
        # Audio is completely unvoiced or silent
        return {
            "has_voicing": False,
            "raw_f0": [],
            "norm_contour": [0.0] * n_points,
            "mean_f0": 0.0,
            "f0_range": (0.0, 0.0)
        }

    valid_f0 = f0[valid_mask]
    mean_f0 = float(np.mean(valid_f0))
    min_f0 = float(np.min(valid_f0))
    max_f0 = float(np.max(valid_f0))

    # Convert to semitones relative to mean (pitch-invariant shape representation)
    # Semitone distance = 12 * log2(f0 / mean_f0)
    semitones = np.full_like(f0, np.nan)
    semitones[valid_mask] = 12.0 * np.log2(valid_f0 / (mean_f0 + 1e-6))

    # Interpolate nan gaps across voiced sections to get smooth curve
    # Only keep the voiced span (first voiced to last voiced frame)
    voiced_indices = np.where(valid_mask)[0]
    start_idx = voiced_indices[0]
    end_idx = voiced_indices[-1]

    sub_semi = semitones[start_idx:end_idx + 1]
    sub_x = np.arange(len(sub_semi))
    known_x = np.where(~np.isnan(sub_semi))[0]
    known_y = sub_semi[known_x]

    if len(known_x) < 2:
        smooth_semi = np.zeros(n_points)
    else:
        # Linear interp inside voiced region
        interp_y = np.interp(sub_x, known_x, known_y)
        # Resample to standard n_points (e.g. 100 points)
        target_grid = np.linspace(0, len(sub_semi) - 1, n_points)
        smooth_semi = np.interp(target_grid, sub_x, interp_y)

    return {
        "has_voicing": True,
        "raw_f0": [round(float(v), 1) if not np.isnan(v) else None for v in f0],
        "norm_contour": [round(float(v), 2) for v in smooth_semi],
        "mean_f0": round(mean_f0, 1),
        "f0_range": (round(min_f0, 1), round(max_f0, 1))
    }

def align_and_score_pitch(ref_contour: List[float], user_contour: List[float]) -> Tuple[float, List[float], List[float]]:
    """
    Perform Dynamic Time Warping (DTW) on normalized pitch contours.
    Returns:
    - Pitch similarity score (0.0 - 100.0)
    - Aligned reference contour
    - Aligned user contour
    """
    ref = np.array(ref_contour, dtype=np.float32)
    usr = np.array(user_contour, dtype=np.float32)

    if len(ref) == 0 or len(usr) == 0 or np.all(usr == 0):
        return 0.0, list(ref), list(usr)

    # Center both to zero mean
    ref_norm = ref - np.mean(ref)
    usr_norm = usr - np.mean(usr)

    # DTW via cumulative cost matrix
    N, M = len(ref_norm), len(usr_norm)
    dist = cdist(ref_norm[:, None], usr_norm[:, None], metric="euclidean")

    D = np.zeros((N + 1, M + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf
    D[1:, 1:] = dist

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            D[i, j] = dist[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # Backtracking path
    i, j = N, M
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = [D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]
        m = min(steps)
        if m == D[i - 1, j - 1]:
            i -= 1
            j -= 1
        elif m == D[i - 1, j]:
            i -= 1
        else:
            j -= 1
    path.reverse()

    aligned_ref = [round(float(ref_norm[p[0]]), 2) for p in path]
    aligned_usr = [round(float(usr_norm[p[1]]), 2) for p in path]

    # Normalized DTW distance
    avg_dist = D[N, M] / len(path)

    # Score: distance 0 -> 100%, distance 3.0+ semitones -> 0%
    score = max(0.0, min(100.0, (1.0 - (avg_dist / 3.0)) * 100.0))

    return round(score, 1), aligned_ref, aligned_usr
