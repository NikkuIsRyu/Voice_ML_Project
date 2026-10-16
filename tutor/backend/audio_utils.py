import io
import os
import shutil
import subprocess
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa
from scipy.spatial.distance import cdist
from scipy import signal
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

def trim_speech(
    y: np.ndarray,
    sr: int = TARGET_SR,
    top_db: float = 28.0,
    frame_length: int = 512,
    hop_length: int = 160,
    pad_ms: float = 40.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Remove leading and trailing silence/clicks from speech audio using bandpass-filtered energy VAD.
    - Filters 70 Hz - 5000 Hz using 4th-order Butterworth to prevent low-frequency rumble / plosives from biasing energy.
    - Measures short-time RMS energy relative to 95th percentile (robust against loud onset clicks/spikes).
    - Preserves pad_ms (default 40ms) margin on both ends.
    - Returns (trimmed_y, vad_meta).
    """
    if y is None or len(y) == 0:
        return np.array([], dtype=np.float32), {
            "speech_ratio": 0.0,
            "trimmed_ms_leading": 0.0,
            "trimmed_ms_trailing": 0.0,
            "active_duration_s": 0.0,
            "start_sample": 0,
            "end_sample": 0
        }

    # Bandpass filter for speech energy estimation (70 Hz to 5000 Hz)
    nyquist = float(sr) / 2.0
    low_cutoff = max(20.0, min(70.0, nyquist * 0.5))
    high_cutoff = max(low_cutoff + 50.0, min(5000.0, nyquist - 100.0))

    try:
        sos = signal.butter(4, [low_cutoff, high_cutoff], btype="bandpass", fs=sr, output="sos")
        filtered = signal.sosfiltfilt(sos, y)
    except Exception:
        filtered = y

    # Calculate frame RMS energy
    rms = librosa.feature.rms(y=filtered, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) == 0:
        return y, {
            "speech_ratio": 1.0,
            "trimmed_ms_leading": 0.0,
            "trimmed_ms_trailing": 0.0,
            "active_duration_s": len(y) / sr,
            "start_sample": 0,
            "end_sample": len(y)
        }

    p95_energy = float(np.percentile(rms, 95))
    if p95_energy < 1e-5:
        # Near complete silence
        return y, {
            "speech_ratio": 0.0,
            "trimmed_ms_leading": 0.0,
            "trimmed_ms_trailing": 0.0,
            "active_duration_s": 0.0,
            "start_sample": 0,
            "end_sample": len(y)
        }

    # Relative energy threshold based on top_db below p95
    thresh = p95_energy * (10.0 ** (-float(top_db) / 20.0))
    is_active = (rms >= thresh)

    # Filter out isolated 1-frame click/spike artifacts (< 20ms)
    sustained_active = np.zeros_like(is_active, dtype=bool)
    n_frames = len(is_active)
    i = 0
    while i < n_frames:
        if is_active[i]:
            j = i
            while j < n_frames and is_active[j]:
                j += 1
            if (j - i) >= 2:
                sustained_active[i:j] = True
            i = j
        else:
            i += 1

    active_indices = np.where(sustained_active)[0]
    if len(active_indices) == 0:
        active_indices = np.where(is_active)[0]

    if len(active_indices) == 0:
        return y, {
            "speech_ratio": 0.0,
            "trimmed_ms_leading": 0.0,
            "trimmed_ms_trailing": 0.0,
            "active_duration_s": 0.0,
            "start_sample": 0,
            "end_sample": len(y)
        }

    start_frame = int(active_indices[0])
    end_frame = int(active_indices[-1])

    start_sample = start_frame * hop_length
    end_sample = min(len(y), end_frame * hop_length + frame_length)

    pad_samples = int((pad_ms / 1000.0) * sr)
    start_sample = max(0, start_sample - pad_samples)
    end_sample = min(len(y), end_sample + pad_samples)

    trimmed_y = y[start_sample:end_sample]
    trimmed_ms_leading = (start_sample / sr) * 1000.0
    trimmed_ms_trailing = ((len(y) - end_sample) / sr) * 1000.0
    active_dur = len(trimmed_y) / sr
    ratio = len(trimmed_y) / max(len(y), 1)

    vad_meta = {
        "speech_ratio": round(ratio, 3),
        "trimmed_ms_leading": round(trimmed_ms_leading, 1),
        "trimmed_ms_trailing": round(trimmed_ms_trailing, 1),
        "active_duration_s": round(active_dur, 3),
        "start_sample": start_sample,
        "end_sample": end_sample
    }

    return trimmed_y, vad_meta

def load_and_resample(
    file_bytes_or_path,
    target_sr: int = TARGET_SR,
    min_duration: float = 0.3,
    min_rms: float = 0.003
) -> Tuple[np.ndarray, int]:
    """
    Load audio from file path or bytes, convert to mono, and resample to target_sr.
    Decodes WebM, OGG, MP4, WAV using ffmpeg subprocess with soundfile fallback.
    Trims edge silence/clicks via VAD before peak-normalizing to preserve speech dynamic range.
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

    # 1. Trim speech with spectral VAD (removes leading/trailing clicks and silence)
    trimmed_data, vad_meta = trim_speech(data, sr=sr)
    if len(trimmed_data) > 0 and vad_meta["active_duration_s"] >= 0.12:
        data = trimmed_data

    # 2. Peak normalization applied AFTER trimming to prevent clicks from depressing speech volume
    max_val = float(np.max(np.abs(data)))
    if max_val > 1e-4:
        data = data / max_val * 0.95

    return data, sr

def extract_f0_contour(audio: np.ndarray, sr: int = TARGET_SR, n_points: int = 100) -> Dict[str, Any]:
    """
    Extract F0 pitch contour using fast librosa.yin algorithm with relative RMS energy-gated voicing
    and edge-burst noise filtering. Converts to semitones, normalizes trajectory, and interpolates to n_points.
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

    # 3-point median filter on raw f0 to eliminate single-frame octave jumps/glitches
    if len(f0) >= 3:
        f0 = signal.medfilt(f0, 3)

    # Valid voiced frames: relative energy gate + bounds check + spectral flatness check
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    sf = librosa.feature.spectral_flatness(y=audio, n_fft=1024, hop_length=256)[0]

    # Ensure length match
    min_len = min(len(f0), len(rms), len(sf))
    f0 = f0[:min_len]
    rms = rms[:min_len]
    sf = sf[:min_len]

    p90_rms = float(np.percentile(rms, 90)) if len(rms) > 0 else 0.012
    rms_thresh = max(0.008, 0.08 * p90_rms)
    # Voiced speech has low spectral flatness (< 0.20) unlike unvoiced fricatives/noise (> 0.35)
    valid_mask = (rms > rms_thresh) & (f0 > fmin * 1.02) & (f0 < fmax * 0.98) & (sf < 0.20)

    # Filter out short isolated voicing bursts (< 3 consecutive frames = < 48ms)
    # to eliminate unvoiced initial consonant clicks/fricative glitches
    cleaned_mask = np.copy(valid_mask)
    n_frames = len(valid_mask)
    i = 0
    while i < n_frames:
        if cleaned_mask[i]:
            j = i
            while j < n_frames and cleaned_mask[j]:
                j += 1
            island_len = j - i
            if island_len < 3:
                cleaned_mask[i:j] = False
            i = j
        else:
            i += 1
    valid_mask = cleaned_mask
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
