import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

DB_PATH = Path(__file__).resolve().parent / "tutor.db"

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # 1. Practice attempts log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS practice_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        item_type TEXT, -- 'vocab' or 'dialogue'
        item_id TEXT,
        target_text TEXT,
        target_pinyin TEXT,
        recognized_text TEXT,
        cer REAL,
        tone_accuracy REAL,
        pitch_score REAL,
        overall_score REAL,
        error_details TEXT, -- JSON breakdown of mismatched tones/characters
        audio_filename TEXT
    )
    """)

    # 2. Spaced Repetition (SRS) items for vocabulary mastery
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS srs_items (
        item_id TEXT PRIMARY KEY,
        text TEXT,
        pinyin TEXT,
        translation TEXT,
        lesson TEXT,
        repetitions INTEGER DEFAULT 0,
        interval_days REAL DEFAULT 1.0,
        ease_factor REAL DEFAULT 2.5,
        next_review_time REAL DEFAULT 0,
        last_score REAL DEFAULT 0,
        total_attempts INTEGER DEFAULT 0,
        mastered INTEGER DEFAULT 0
    )
    """)

    # 3. Dialogue role-play progress
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dialogue_progress (
        dialogue_id TEXT PRIMARY KEY,
        title TEXT,
        total_lines INTEGER,
        completed_turns INTEGER DEFAULT 0,
        average_score REAL DEFAULT 0,
        last_practiced REAL DEFAULT 0
    )
    """)

    # 4. Pronunciation & Tone Error Tracking
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pron_errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        syllable TEXT,
        expected_tone INTEGER,
        heard_tone INTEGER,
        issue TEXT,
        comment TEXT
    )
    """)

    conn.commit()
    conn.close()

def log_practice(item_type: str, item_id: str, target_text: str, target_pinyin: str,
                 recognized_text: str, cer: float, tone_accuracy: float, pitch_score: float,
                 overall_score: float, error_details: Dict[str, Any], audio_filename: str = ""):
    conn = get_db()
    cursor = conn.cursor()
    now = time.time()
    cursor.execute("""
    INSERT INTO practice_records (
        timestamp, item_type, item_id, target_text, target_pinyin, recognized_text,
        cer, tone_accuracy, pitch_score, overall_score, error_details, audio_filename
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now, item_type, item_id, target_text, target_pinyin, recognized_text,
        cer, tone_accuracy, pitch_score, overall_score, json.dumps(error_details, ensure_ascii=False), audio_filename
    ))

    # If vocab item, update SRS status
    if item_type == "vocab":
        update_srs(cursor, item_id, overall_score, now)

    conn.commit()
    conn.close()

def update_srs(cursor, item_id: str, score: float, now: float):
    cursor.execute("SELECT repetitions, interval_days, ease_factor FROM srs_items WHERE item_id = ?", (item_id,))
    row = cursor.fetchone()
    if not row:
        return

    reps = row["repetitions"]
    interval = row["interval_days"]
    ease = row["ease_factor"]

    # Convert score (0 - 100) to SM-2 quality grade (0 - 5)
    if score >= 90:
        grade = 5
    elif score >= 80:
        grade = 4
    elif score >= 65:
        grade = 3
    elif score >= 45:
        grade = 2
    elif score >= 25:
        grade = 1
    else:
        grade = 0

    if grade >= 3:
        if reps == 0:
            interval = 1.0
        elif reps == 1:
            interval = 6.0
        else:
            interval = interval * ease
        reps += 1
    else:
        reps = 0
        interval = 1.0

    ease = ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    ease = max(1.3, ease)

    next_review = now + interval * 86400
    mastered = 1 if (reps >= 3 and grade >= 4) else 0

    cursor.execute("""
    UPDATE srs_items
    SET repetitions = ?, interval_days = ?, ease_factor = ?, next_review_time = ?,
        last_score = ?, total_attempts = total_attempts + 1, mastered = ?
    WHERE item_id = ?
    """, (reps, interval, ease, next_review, score, mastered, item_id))

def get_stats():
    conn = get_db()
    cursor = conn.cursor()
    now = time.time()

    cursor.execute("SELECT COUNT(*) as total FROM srs_items")
    total_words = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) as mastered FROM srs_items WHERE mastered = 1")
    mastered_words = cursor.fetchone()["mastered"]

    cursor.execute("SELECT COUNT(*) as due FROM srs_items WHERE next_review_time <= ? AND total_attempts > 0", (now,))
    due_words = cursor.fetchone()["due"]

    cursor.execute("SELECT COUNT(*) as unpracticed FROM srs_items WHERE total_attempts = 0")
    unpracticed_words = cursor.fetchone()["unpracticed"]

    cursor.execute("SELECT AVG(overall_score) as avg_score, AVG(tone_accuracy) as avg_tone, AVG(pitch_score) as avg_pitch FROM practice_records")
    rec = cursor.fetchone()
    avg_score = rec["avg_score"] or 0.0
    avg_tone = rec["avg_tone"] or 0.0
    avg_pitch = rec["avg_pitch"] or 0.0

    cursor.execute("SELECT COUNT(*) as total_practices FROM practice_records")
    total_practices = cursor.fetchone()["total_practices"]

    # Recent attempts
    cursor.execute("""
    SELECT timestamp, item_type, target_text, target_pinyin, recognized_text, overall_score, tone_accuracy
    FROM practice_records ORDER BY id DESC LIMIT 10
    """)
    recent = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return {
        "total_words": total_words,
        "mastered_words": mastered_words,
        "due_words": due_words,
        "unpracticed_words": unpracticed_words,
        "total_practices": total_practices,
        "average_score": round(avg_score, 1),
        "average_tone_accuracy": round(avg_tone, 1),
        "average_pitch_accuracy": round(avg_pitch, 1),
        "recent_attempts": recent
    }

def log_pron_error(syllable: str, expected_tone: int, heard_tone: int, issue: str, comment: str = ""):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO pron_errors (timestamp, syllable, expected_tone, heard_tone, issue, comment)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (time.time(), syllable, expected_tone, heard_tone, issue, comment))
    conn.commit()
    conn.close()

def get_due_srs_words(limit: int = 25) -> List[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    now = time.time()
    cursor.execute("""
    SELECT item_id as id, text as chinese, pinyin, translation, lesson, repetitions, last_score
    FROM srs_items
    WHERE (next_review_time <= ? AND total_attempts > 0) OR total_attempts = 0
    ORDER BY (CASE WHEN total_attempts > 0 THEN 0 ELSE 1 END), next_review_time ASC
    LIMIT ?
    """, (now, limit))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_pron_weakness_profile() -> Dict[str, Any]:
    conn = get_db()
    cursor = conn.cursor()
    # Tone confusion matrix: (expected_tone, heard_tone) -> count
    cursor.execute("""
    SELECT expected_tone, heard_tone, COUNT(*) as cnt
    FROM pron_errors
    WHERE issue = 'tone'
    GROUP BY expected_tone, heard_tone
    """)
    confusions = [
        {
            "expected_tone": r["expected_tone"],
            "heard_tone": r["heard_tone"],
            "expected": r["expected_tone"],
            "heard": r["heard_tone"],
            "count": r["cnt"]
        }
        for r in cursor.fetchall()
    ]

    # Top problem syllables
    cursor.execute("""
    SELECT syllable, issue, COUNT(*) as cnt
    FROM pron_errors
    GROUP BY syllable, issue
    ORDER BY cnt DESC
    LIMIT 10
    """)
    top_problems = [{"syllable": r["syllable"], "issue": r["issue"], "count": r["cnt"]} for r in cursor.fetchall()]

    conn.close()
    return {
        "tone_confusions": confusions,
        "top_tone_confusions": confusions,
        "top_problem_syllables": top_problems
    }

def log_dialogue_turn(dialogue_id: str, line_index: int, score: float, recognized_text: str):
    conn = get_db()
    cursor = conn.cursor()
    now = time.time()
    cursor.execute("""
    INSERT INTO dialogue_progress (dialogue_id, title, total_lines, completed_turns, average_score, last_practiced)
    VALUES (?, ?, 10, 1, ?, ?)
    ON CONFLICT(dialogue_id) DO UPDATE SET
        completed_turns = completed_turns + 1,
        average_score = (average_score * completed_turns + excluded.average_score) / (completed_turns + 1),
        last_practiced = excluded.last_practiced
    """, (dialogue_id, dialogue_id, score, now))
    conn.commit()
    conn.close()

init_db()

