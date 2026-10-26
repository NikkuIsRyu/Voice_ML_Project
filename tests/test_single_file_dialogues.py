"""
Tests for Improvement 3.9: Single-file multi-speaker dialogues and alignment timestamps.
"""

import json
from pathlib import Path
import pytest
import soundfile as sf
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIA_DIR = PROJECT_ROOT / "output_v2" / "dialogues"
DIA_SLOW_DIR = PROJECT_ROOT / "output_v2_slow" / "dialogues"

EXPECTED_DIALOGUES = [
    "Dialogue2_1", "Dialogue2_2", "Dialogue3_1", "Dialogue3_2", "Dialogue4_1", "Dialogue4_2",
    "Dialogue5_1", "Dialogue5_2", "Dialogue6_1", "Dialogue6_2", "Dialogue7_1", "Dialogue7_2"
]

class TestSingleFileDialogues:
    @pytest.mark.parametrize("dia_id", EXPECTED_DIALOGUES)
    def test_dialogue_single_file_and_timestamps_exist(self, dia_id):
        wav_path = DIA_DIR / f"{dia_id}.wav"
        json_path = DIA_DIR / f"{dia_id}_timestamps.json"

        assert wav_path.exists(), f"Missing dialogue audio: {wav_path}"
        assert json_path.exists(), f"Missing dialogue timestamps: {json_path}"

        # Verify audio properties
        data, sr = sf.read(str(wav_path))
        assert sr == 32000
        assert len(data) > 0
        assert not np.isnan(data).any()
        assert not np.isinf(data).any()
        peak = float(np.max(np.abs(data)))
        assert 0.1 <= peak <= 1.0, f"Abnormal peak amplitude {peak}"

        # Verify timestamp structure
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["dialogue_id"] == dia_id
        assert meta["total_duration_s"] > 5.0
        assert len(meta["turns"]) >= 4

        # Verify chronological order
        prev_end = 0.0
        for turn in meta["turns"]:
            assert turn["start_ms"] >= prev_end
            assert turn["end_ms"] > turn["start_ms"]
            assert len(turn["text"]) > 0
            assert turn["voice"] in ["female_v3", "female_v4", "male_v1", "male_zs"]
            prev_end = turn["end_ms"]

    def test_slow_dialogue_exists(self):
        wav_slow = DIA_SLOW_DIR / "Dialogue2_1.wav"
        assert wav_slow.exists()
        data, sr = sf.read(str(wav_slow))
        assert len(data) > 0
