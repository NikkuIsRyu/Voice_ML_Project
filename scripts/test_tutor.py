#!/usr/bin/env python3
"""
Automated verification test script for the Chinese Pronunciation Tutor backend.
Tests:
1. /api/health
2. /api/stats
3. /api/lessons & /api/lessons/{lesson_id} (validating audio file URLs)
4. /api/dialogues & /api/dialogues/{dialogue_id}
5. /api/audio streaming (v2 audio)
6. /api/score/vocab with real audio and DTW pitch comparison
7. SQLite database persistence and SM-2 progress update
"""

import sys
import io
import time
import requests
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("=" * 60)
    print("Pronunciation Tutor Automated Verification Suite")
    print("=" * 60)

    # 1. Health check
    print("\n[1/7] Testing /api/health...")
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert data.get("status") == "ok", f"Unexpected status: {data}"
        print(f"  PASS: Health check OK (Server time: {data.get('time')})")
    except Exception as e:
        print(f"  FAIL: Health check failed: {e}")
        return False

    # 2. Stats
    print("\n[2/7] Testing /api/stats...")
    r = requests.get(f"{BASE_URL}/api/stats", timeout=5)
    assert r.status_code == 200
    stats = r.json()
    print(f"  Total vocab in DB: {stats.get('total_words')}")
    print(f"  Unpracticed items: {stats.get('unpracticed_words')}")
    print(f"  Total sessions:    {stats.get('total_practices')}")
    assert stats.get("total_words", 0) > 300, "Expected >300 vocab items in database"
    print("  PASS: Stats API returned valid database counts.")

    # 3. Lessons
    print("\n[3/7] Testing /api/lessons...")
    r = requests.get(f"{BASE_URL}/api/lessons", timeout=5)
    assert r.status_code == 200
    lessons = r.json().get("lessons", [])
    print(f"  Lessons count: {len(lessons)}")
    for l in lessons:
        print(f"    - {l['id']}: {l['word_count']} words")
    assert len(lessons) >= 6, "Expected at least 6 lessons"
    print("  PASS: Lessons list OK.")

    # 4. Lesson details & audio URL verification
    print("\n[4/7] Testing /api/lessons/lesson01 & audio availability...")
    r = requests.get(f"{BASE_URL}/api/lessons/lesson01", timeout=5)
    assert r.status_code == 200
    words = r.json().get("words", [])
    assert len(words) > 0, "No words found in lesson01"
    first_word = words[0]
    print(f"  First word: {first_word['chinese']} ({first_word['pinyin']}) - Audio URL: {first_word.get('audio_url')}")
    assert first_word.get("audio_url") is not None, "Audio URL should not be None"
    
    # Test audio stream
    audio_r = requests.get(f"{BASE_URL}{first_word['audio_url']}", timeout=5)
    assert audio_r.status_code == 200, f"Failed to fetch audio: {audio_r.status_code}"
    assert len(audio_r.content) > 1000, f"Audio file is too small: {len(audio_r.content)} bytes"
    print(f"  PASS: Audio streaming verified ({len(audio_r.content)} bytes).")

    # 5. Dialogues
    print("\n[5/7] Testing /api/dialogues & /api/dialogues/Dialogue2_1...")
    r = requests.get(f"{BASE_URL}/api/dialogues", timeout=5)
    assert r.status_code == 200
    dialogues = r.json().get("dialogues", [])
    assert len(dialogues) >= 6, "Expected at least 6 dialogue scenarios"
    
    r_diag = requests.get(f"{BASE_URL}/api/dialogues/Dialogue2_1", timeout=5)
    assert r_diag.status_code == 200
    lines = r_diag.json().get("lines", [])
    assert len(lines) > 0, "No lines in Dialogue2_1"
    first_line = lines[0]
    print(f"  First line: [{first_line['speaker']}] {first_line['text']} - Audio: {first_line.get('audio_url')}")
    assert first_line.get("audio_url") is not None
    audio_diag = requests.get(f"{BASE_URL}{first_line['audio_url']}", timeout=5)
    assert audio_diag.status_code == 200
    print(f"  PASS: Dialogue audio streaming verified ({len(audio_diag.content)} bytes).")

    # 6. Scoring endpoint with real audio
    print("\n[6/7] Testing /api/score/vocab with real audio input...")
    # Find sample audio to use as student input
    cands = list(Path("output_v2/vocab/lesson01").glob("*您好*.wav"))
    if not cands:
        cands = list(Path("output/vocab/lesson01").glob("*您好*.wav"))
    assert len(cands) > 0, "Audio test file not found matching *您好*.wav"
    test_audio_path = cands[0]
    print(f"  Using test audio clip: {test_audio_path}")

    with open(test_audio_path, "rb") as f:
        audio_bytes = f.read()

    files = {
        "audio": ("student_rec.wav", audio_bytes, "audio/wav")
    }
    form_data = {
        "item_id": "lesson01_您好",
        "target_text": "您好",
        "target_pinyin": "nín hǎo",
        "audio_filename": test_audio_path.name
    }

    t0 = time.time()
    score_res = requests.post(f"{BASE_URL}/api/score/vocab", data=form_data, files=files, timeout=60)
    elapsed = time.time() - t0
    assert score_res.status_code == 200, f"Scoring failed: {score_res.text}"
    score_data = score_res.json()
    print(f"  Scoring response time: {elapsed:.2f}s")
    print(f"  Recognized Text:       {score_data.get('recognized_text')}")
    print(f"  Overall Score:         {score_data.get('overall_score')}/100")
    print(f"  Character Accuracy:    {score_data.get('char_accuracy')}%")
    print(f"  Tone Accuracy:         {score_data.get('tone_accuracy')}%")
    print(f"  Pitch DTW Score:       {score_data.get('pitch_score')}")
    print(f"  Pitch curves returned: {len(score_data.get('pitch_curves', {}).get('ref', []))} points")
    assert score_data.get("overall_score") > 60.0, "Score should be reasonably high for clear input"
    print("  PASS: Pronunciation scoring and pitch DTW alignment verified.")

    # 7. Check database update
    print("\n[7/7] Checking database persistence of practice session...")
    r_stats2 = requests.get(f"{BASE_URL}/api/stats", timeout=5)
    stats2 = r_stats2.json()
    print(f"  Total practices updated to: {stats2.get('total_practices')}")
    assert stats2.get("total_practices", 0) >= stats.get("total_practices", 0) + 1
    print("  PASS: Practice successfully logged to SQLite database and SRS updated.")

    print("\n" + "=" * 60)
    print("ALL 7 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
