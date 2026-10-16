"""
Unit tests for VAD filtering, silence/click stripping, and F0 contour stabilization.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tutor" / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from audio_utils import trim_speech, extract_f0_contour, align_and_score_pitch

def generate_tone_signal(freq_hz: float = 240.0, duration_s: float = 0.6, sr: int = 16000, amplitude: float = 0.4) -> np.ndarray:
    """Generate a clean harmonic voiced signal with smooth raised-cosine envelope."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Fundamental + 2 harmonics to simulate natural voice
    signal = (
        amplitude * np.sin(2 * np.pi * freq_hz * t)
        + 0.5 * amplitude * np.sin(2 * np.pi * 2 * freq_hz * t)
        + 0.25 * amplitude * np.sin(2 * np.pi * 3 * freq_hz * t)
    )
    # Envelope (fade in/out 20ms)
    fade_len = int(0.02 * sr)
    env = np.ones_like(signal)
    env[:fade_len] = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
    env[-fade_len:] = 0.5 * (1.0 - np.cos(np.linspace(np.pi, 0, fade_len)))
    return (signal * env).astype(np.float32)

class TestVADAndContourExtraction:
    def test_vad_strips_leading_trailing_silence_and_clicks(self):
        sr = 16000
        clean_speech = generate_tone_signal(freq_hz=250.0, duration_s=0.6, sr=sr, amplitude=0.4)

        # Create contaminated audio: 1.2s silence + loud 3ms click + clean speech + 0.8s silence
        silence_lead = np.zeros(int(1.2 * sr), dtype=np.float32)
        click = np.zeros(int(0.003 * sr), dtype=np.float32)
        click[:5] = [0.95, -0.9, 0.85, -0.8, 0.5]  # sharp microphone click impulse
        silence_trail = np.zeros(int(0.8 * sr), dtype=np.float32)

        contaminated = np.concatenate([silence_lead, click, clean_speech, silence_trail])

        trimmed_clean, clean_meta = trim_speech(clean_speech, sr=sr)
        trimmed_contam, contam_meta = trim_speech(contaminated, sr=sr)

        # Verification: VAD must strip away the majority of the leading and trailing silence
        assert contam_meta["trimmed_ms_leading"] >= 1000.0, f"Expected >= 1000ms leading trim, got {contam_meta['trimmed_ms_leading']}"
        assert contam_meta["trimmed_ms_trailing"] >= 600.0, f"Expected >= 600ms trailing trim, got {contam_meta['trimmed_ms_trailing']}"
        assert abs(contam_meta["active_duration_s"] - 0.6) < 0.15

        # DTW comparison between clean pitch contour and trimmed contaminated audio contour
        clean_f0 = extract_f0_contour(trimmed_clean, sr=sr, n_points=80)
        contam_f0 = extract_f0_contour(trimmed_contam, sr=sr, n_points=80)

        assert clean_f0["has_voicing"] is True
        assert contam_f0["has_voicing"] is True

        score, aligned_clean, aligned_contam = align_and_score_pitch(
            clean_f0["norm_contour"],
            contam_f0["norm_contour"]
        )

        # DTW pitch similarity must be virtually identical (> 95%)
        assert score >= 95.0, f"Expected DTW score >= 95.0, got {score}"

    def test_unvoiced_initial_stability(self):
        sr = 16000
        # 120ms unvoiced burst / fricative noise (simulating 'sh' / 'x' or breath plosive)
        noise_len = int(0.12 * sr)
        np.random.seed(42)
        noise = (np.random.randn(noise_len) * 0.15).astype(np.float32)

        # Followed by 500ms voiced vowel (Tone 1, 230 Hz)
        voiced = generate_tone_signal(freq_hz=230.0, duration_s=0.5, sr=sr, amplitude=0.45)
        combined = np.concatenate([noise, voiced])

        f0_info = extract_f0_contour(combined, sr=sr, n_points=80)
        assert f0_info["has_voicing"] is True

        # Mean F0 should be close to 230 Hz (+/- 15 Hz) without octave jumps from initial noise
        assert abs(f0_info["mean_f0"] - 230.0) < 15.0, f"Mean F0 skewed to {f0_info['mean_f0']}"

        # Standard deviation of normalized contour on Tone 1 should be low (flat pitch)
        norm_curve = np.array(f0_info["norm_contour"])
        std_semitones = float(np.std(norm_curve))
        assert std_semitones < 1.8, f"Normalized pitch contour too erratic for Tone 1: std={std_semitones}"

    def test_vad_metadata_structure(self):
        sr = 16000
        dummy_audio = generate_tone_signal(freq_hz=200.0, duration_s=0.4, sr=sr)
        padded_audio = np.concatenate([np.zeros(sr // 2, dtype=np.float32), dummy_audio, np.zeros(sr // 2, dtype=np.float32)])

        _, meta = trim_speech(padded_audio, sr=sr)
        expected_keys = {"speech_ratio", "trimmed_ms_leading", "trimmed_ms_trailing", "active_duration_s", "start_sample", "end_sample"}
        assert expected_keys.issubset(meta.keys())
        assert meta["active_duration_s"] > 0.3
        assert meta["trimmed_ms_leading"] > 400.0
