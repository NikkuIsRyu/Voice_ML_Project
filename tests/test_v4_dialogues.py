"""
Unit tests for v4 dialogues: schema parsing, speaker mapping, and pinyin generation.
"""

import sys
from pathlib import Path
import pytest
import yaml
from pypinyin import pinyin, Style

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_multi_speaker_dialogues import parse_dialogue_file

ALL_V4_DIALOGUES = [
    "Dialogue2_1", "Dialogue2_2",
    "Dialogue3_1", "Dialogue3_2",
    "Dialogue4_1", "Dialogue4_2",
    "Dialogue5_1", "Dialogue5_2",
    "Dialogue6_1", "Dialogue6_2",
    "Dialogue7_1", "Dialogue7_2"
]

@pytest.fixture(scope="module")
def speaker_cfg():
    with open(PROJECT_ROOT / "configs" / "dialogue_speakers.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def voice_cfg():
    with open(PROJECT_ROOT / "configs" / "voice_profiles.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["voices"]

@pytest.mark.parametrize("dia_id", ALL_V4_DIALOGUES)
def test_dialogue_file_exists_and_parses(dia_id, speaker_cfg, voice_cfg):
    dia_path = PROJECT_ROOT / "data" / "dialogues" / f"{dia_id}.txt"
    assert dia_path.exists(), f"Missing dialogue file: {dia_path}"

    turns = parse_dialogue_file(dia_path, speaker_cfg)
    assert 6 <= len(turns) <= 16, f"Unexpected turn count for {dia_id}: {len(turns)}"

    for turn in turns:
        assert turn["dialogue_id"] == dia_id
        assert turn["speaker_label"] in ["A", "B", "C"]
        # Voice must be defined in voice profiles (or active alias)
        assert turn["voice_profile"] in voice_cfg or turn["voice_profile"] in ["female_v3", "female_v4", "male_v1", "male_v2", "male_zs", "male_b_v1"]
        assert len(turn["text"]) > 2

        # Pinyin validation
        py_list = pinyin(turn["text"], style=Style.TONE)
        assert len(py_list) > 0
        py_str = " ".join([item[0] for item in py_list])
        assert len(py_str.strip()) > 0
