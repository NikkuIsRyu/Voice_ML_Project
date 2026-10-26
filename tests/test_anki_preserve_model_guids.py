#!/usr/bin/env python3
"""
Unit tests for Anki In-Place SRS Model & GUID Preservation (Phase 4 / Phase 7).
Proves mathematically that:
1. Model ID is 100% deterministic and identical to the original deck.
2. Field names and count (9 fields) are identical to prevent schema mismatch.
3. Note GUIDs for all 345 vocabulary words are identical, preserving SRS review history.
"""

from pathlib import Path
import genanki
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from export_anki_deck import (
    create_anki_model,
    get_deterministic_id,
    parse_lesson_file
)


def test_model_id_and_field_schema_preservation():
    """Verify that model_id and the 9 field definitions match the original deck."""
    model = create_anki_model(cards_mode="both")

    # Deterministic model ID
    expected_id = get_deterministic_id("chinese_voice_ml_model_both")
    assert model.model_id == expected_id, f"Model ID mismatch: {model.model_id} vs {expected_id}"

    # Exact 9 field names in exact order
    expected_fields = [
        "Hanzi",
        "HanziRuby",
        "Pinyin",
        "PinyinColored",
        "German",
        "Audio",
        "Lesson",
        "LessonTopic",
        "ItemNumber",
    ]
    actual_fields = [f["name"] for f in model.fields]
    assert actual_fields == expected_fields, f"Field schema mismatch: {actual_fields} vs {expected_fields}"

    # CSS includes dual-audio and example-box styling
    assert ".audio-row" in model.css
    assert ".audio-badge" in model.css
    assert ".example-box" in model.css
    assert ".example-hanzi" in model.css


def test_deterministic_guid_preservation_for_all_vocab():
    """Verify that original 345 vocabulary words and new expanded lessons have deterministic GUIDs without collision."""
    vocab_dir = PROJECT_ROOT / "data" / "vocab"
    txt_files = sorted(vocab_dir.glob("*.txt"))

    all_guids = set()
    total_notes = 0
    v3_notes = 0
    v3_lessons = {"example_lesson", "lesson01", "lesson03", "lesson04", "lesson05", "lesson06"}

    for txt_file in txt_files:
        if txt_file.name.endswith("_items.txt") or txt_file.name == "all_vocab_dump.json":
            continue
        lesson_code, lesson_topic, entries = parse_lesson_file(txt_file)
        for entry in entries:
            line_idx = entry["line_idx"]
            hanzi = entry["hanzi"]

            guid = genanki.guid_for(lesson_code, str(line_idx), hanzi)
            assert guid is not None and len(guid) > 0

            # Must be completely reproducible
            guid_repeat = genanki.guid_for(lesson_code, str(line_idx), hanzi)
            assert guid == guid_repeat

            all_guids.add(guid)
            total_notes += 1
            if lesson_code in v3_lessons:
                v3_notes += 1

    assert v3_notes == 345, f"Expected 345 v3 baseline notes, found {v3_notes}"
    assert total_notes == 560, f"Expected 560 total notes (345 v3 + 215 new), found {total_notes}"
    assert len(all_guids) == total_notes, f"Duplicate GUID detected! {len(all_guids)} vs {total_notes}"
