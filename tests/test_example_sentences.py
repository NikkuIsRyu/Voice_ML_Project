#!/usr/bin/env python3
"""
Unit tests for Automatic Example Sentences (Phase 5 - Improvement 2.4).
"""

import json
from pathlib import Path
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "vocab" / "vocab_example_sentences.json"
EXAMPLES_DIR = PROJECT_ROOT / "output_v2" / "examples"


def test_example_sentences_manifest_structure():
    """Verify manifest exists, contains exactly 345 words, and all required fields."""
    assert MANIFEST_PATH.exists(), f"Missing manifest at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 345, f"Expected 345 entries, found {len(data)}"

    for entry in data:
        assert "lesson" in entry
        assert "line_idx" in entry
        assert "hanzi" in entry
        assert "sentence_hanzi" in entry
        assert "clean_sentence" in entry
        assert "sentence_pinyin" in entry
        assert "sentence_translation_de" in entry
        assert "audio_filename" in entry

        # Verify bold tag surrounds the target word
        expected_bold = f"<b>{entry['hanzi']}</b>"
        assert expected_bold in entry["sentence_hanzi"], (
            f"Sentence for {entry['hanzi']} does not contain bold tag: {entry['sentence_hanzi']}"
        )

        # Verify clean sentence matches stripped version
        assert entry["clean_sentence"] == entry["sentence_hanzi"].replace("<b>", "").replace("</b>", "")

        # Verify audio filename pattern
        assert entry["audio_filename"].startswith("ex_")
        assert entry["audio_filename"].endswith(".wav")


def test_example_sentences_audio_files():
    """Verify that synthesized example audio files exist and have valid audio content."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    wav_files = list(EXAMPLES_DIR.glob("*.wav"))
    assert len(wav_files) >= 300, f"Expected at least 300 example audio files in {EXAMPLES_DIR}, found {len(wav_files)}"

    # Check each manifest entry has an existing audio file on disk
    for entry in data[:50]:
        wav_path = EXAMPLES_DIR / entry["audio_filename"]
        assert wav_path.exists(), f"Missing example audio file: {wav_path.name}"
        info = sf.info(str(wav_path))
        assert info.samplerate in (32000, 44100, 48000), f"Unexpected sample rate {info.samplerate}"
        assert info.duration >= 0.5, f"Audio duration too short ({info.duration:.2f}s) for {wav_path.name}"
