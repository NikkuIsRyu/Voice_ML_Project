"""
Unit and Integration Tests for Chinese Pronunciation Tutor & Pronunciation Engine.

Tests cover Phase A1 through A7:
- A1: Browser audio decoding (WebM Opus, OGG, MP4 AAC, WAV) and silence/duration validation
- A4: Sandhi compliance (你好 -> T2-T3, 不是 -> T2-T4)
- A4: Negative minimal-pair tone error detection (买 vs 卖, 妈 vs 骂, 是 vs 十) >= 80% pass
- A3: Latency benchmark (p50 <= 3s)
- A2/A5: FastAPI endpoints return valid numbers, no undefined/null
"""

import os
import sys
import time
import pytest
import numpy as np
from pathlib import Path

# Set up project root and Windows CUDA library paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tutor" / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows CTranslate2 and PyTorch CUDA DLLs are found
torch_lib = Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib"
if torch_lib.exists():
    os.environ["PATH"] = str(torch_lib) + ";" + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

from audio_utils import load_and_resample, extract_f0_contour, align_and_score_pitch
from sandhi import apply_tone_sandhi
from pronunciation_engine import get_pronunciation_engine
from scoring import PronunciationScorer
from fastapi.testclient import TestClient
from main import app

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

# ----------------- A1: Audio Decoding & Validation -----------------

class TestAudioDecoding:
    @pytest.mark.parametrize("fmt,fname", [
        ("WAV", "nihao.wav"),
        ("WebM", "nihao.webm"),
        ("OGG", "nihao.ogg"),
        ("MP4", "nihao.mp4")
    ])
    def test_decode_browser_formats(self, fmt, fname):
        fpath = FIXTURES_DIR / fname
        assert fpath.exists(), f"Missing fixture {fname}"
        audio, sr = load_and_resample(fpath)
        assert sr == 16000
        assert len(audio) > 0
        assert audio.ndim == 1
        assert np.max(np.abs(audio)) > 0.05

    def test_reject_silent_audio(self):
        silent_f = FIXTURES_DIR / "silent.webm"
        with pytest.raises(ValueError) as excinfo:
            load_and_resample(silent_f, min_duration=0.3)
        assert "too short" in str(excinfo.value) or "too quiet" in str(excinfo.value)

# ----------------- A4: Sandhi Compliance -----------------

class TestSandhiRules:
    def test_3_3_sandhi(self):
        syls = [{"char": "你", "tone": 3}, {"char": "好", "tone": 3}]
        res = apply_tone_sandhi(syls)
        assert [x["expected_tone"] for x in res] == [2, 3]

    def test_bu_sandhi(self):
        syls = [{"char": "不", "tone": 4}, {"char": "是", "tone": 4}]
        res = apply_tone_sandhi(syls)
        assert [x["expected_tone"] for x in res] == [2, 4]

        syls2 = [{"char": "不", "tone": 4}, {"char": "好", "tone": 3}]
        res2 = apply_tone_sandhi(syls2)
        assert [x["expected_tone"] for x in res2] == [4, 3]

    def test_yi_sandhi(self):
        syls = [{"char": "一", "tone": 1}, {"char": "个", "tone": 4}]
        assert [x["expected_tone"] for x in apply_tone_sandhi(syls)] == [2, 4]

        syls2 = [{"char": "一", "tone": 1}, {"char": "天", "tone": 1}]
        assert [x["expected_tone"] for x in apply_tone_sandhi(syls2)] == [4, 1]

        syls3 = [{"char": "第", "tone": 4}, {"char": "一", "tone": 1}]
        assert [x["expected_tone"] for x in apply_tone_sandhi(syls3)] == [4, 1]

# ----------------- A4: Negative Minimal-Pair Tone Tests -----------------

class TestMinimalPairToneDiscrimination:
    @pytest.fixture(scope="class")
    def engine(self):
        return get_pronunciation_engine()

    def test_minimal_pairs_detection(self, engine):
        """
        Evaluate audio clips against mismatched tone targets.
        Engine must detect tone errors on >= 80% of pairs.
        """
        pairs = [
            # (audio_file, target_hanzi, true_tone, expected_target_tone)
            ("mai4_csmsc.wav", "买", 4, 3),   # Submitted 卖(T4) against target 买(T3)
            ("mai3_csmsc.wav", "卖", 3, 4),   # Submitted 买(T3) against target 卖(T4)
            ("ma4_csmsc.wav", "妈", 4, 1),    # Submitted 骂(T4) against target 妈(T1)
            ("ma1_csmsc.wav", "骂", 1, 4),    # Submitted 妈(T1) against target 骂(T4)
            ("shi4_v2.wav", "十", 4, 2),      # Submitted 是(T4) against target 十(T2)
            ("shi2_csmsc.wav", "是", 2, 4),   # Submitted 十(T2) against target 是(T4)
        ]

        detected_count = 0
        total = len(pairs)

        for audio_fname, target_char, true_tone, exp_tone in pairs:
            fpath = FIXTURES_DIR / audio_fname
            assert fpath.exists(), f"Fixture missing: {audio_fname}"
            audio, sr = load_and_resample(fpath)
            res = engine.evaluate(audio, sr, target_char)

            # Check if any syllable reported a tone issue or heard_tone != exp_tone
            tone_error_flagged = False
            for syl in res["syllables"]:
                if syl.get("issue") == "tone" or (syl.get("heard_tone") is not None and syl.get("heard_tone") != exp_tone):
                    tone_error_flagged = True
                    break

            if tone_error_flagged or res["tone_accuracy"] < 100.0:
                detected_count += 1
                print(f"PASS: {audio_fname} against target '{target_char}' flagged tone error (Tone acc: {res['tone_accuracy']}%)")
            else:
                print(f"FAIL: {audio_fname} against target '{target_char}' NOT flagged as tone error")

        detection_rate = detected_count / total
        print(f"Minimal pairs detection rate: {detection_rate:.1%} ({detected_count}/{total})")
        assert detection_rate >= 0.80, f"Detection rate {detection_rate:.1%} is below required 80% threshold"

# ----------------- A3: Latency Benchmark -----------------

class TestLatencyBenchmark:
    def test_2_syllable_scoring_latency(self):
        engine = get_pronunciation_engine()
        audio, sr = load_and_resample(FIXTURES_DIR / "nihao.wav")

        latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = engine.evaluate(audio, sr, "你好")
            dt = time.perf_counter() - t0
            latencies.append(dt)

        p50 = float(np.median(latencies))
        print(f"Engine p50 latency for '你好': {p50:.3f} s")
        assert p50 <= 3.0, f"p50 latency {p50:.3f}s exceeded target 3.0s"

# ----------------- A2/A5: API Endpoints (Integration) -----------------

class TestApiEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)

    def test_vocab_scoring_endpoint_webm(self, client):
        webm_path = FIXTURES_DIR / "nihao.webm"
        with open(webm_path, "rb") as f:
            resp = client.post(
                "/api/score/vocab",
                files={"audio": ("user_rec.webm", f, "audio/webm")},
                data={
                    "item_id": "test_001",
                    "target_text": "你好",
                    "target_pinyin": "nǐ hǎo",
                    "audio_filename": "example_lesson_005_你好.wav"
                }
            )

        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.json()
        assert isinstance(data["overall_score"], (int, float))
        assert isinstance(data["tone_accuracy"], (int, float))
        assert isinstance(data["char_accuracy"], (int, float))
        assert isinstance(data["pitch_score"], (int, float))
        assert len(data["syllables"]) == 2
        assert "aligned_ref" in data["pitch_curves"]
        assert "aligned_user" in data["pitch_curves"]

    def test_vocab_scoring_silent_error_banner(self, client):
        silent_path = FIXTURES_DIR / "silent.webm"
        with open(silent_path, "rb") as f:
            resp = client.post(
                "/api/score/vocab",
                files={"audio": ("user_rec.webm", f, "audio/webm")},
                data={
                    "item_id": "test_silent",
                    "target_text": "你好"
                }
            )

        assert resp.status_code == 422
        err = resp.json()
        assert "detail" in err
        assert "too short" in err["detail"].lower() or "too quiet" in err["detail"].lower()

    def test_stats_weakness_endpoint(self, client):
        resp = client.get("/api/stats/weakness")
        assert resp.status_code == 200
        data = resp.json()
        assert "top_tone_confusions" in data
        assert "top_problem_syllables" in data

    def test_srs_due_endpoint(self, client):
        resp = client.get("/api/srs/due")
        assert resp.status_code == 200
        data = resp.json()
        assert "words" in data
