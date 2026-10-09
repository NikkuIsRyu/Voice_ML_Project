import sqlite3
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "tutor" / "data" / "tutor.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_talk_tables():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS talk_sessions (
        id TEXT PRIMARY KEY,
        mode TEXT NOT NULL,
        scenario TEXT,
        started_at REAL NOT NULL,
        ended_at REAL,
        summary_json TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS talk_turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        turn_idx INTEGER NOT NULL,
        role TEXT NOT NULL,
        text_zh TEXT NOT NULL,
        text_pinyin TEXT,
        text_translation TEXT,
        audio_path TEXT,
        pron_report_json TEXT,
        corrections_json TEXT,
        latency_ms REAL,
        created_at REAL NOT NULL,
        FOREIGN KEY (session_id) REFERENCES talk_sessions(id)
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_talk_turns_session ON talk_turns(session_id)")
    conn.commit()
    conn.close()

# Auto-initialize on module load
init_talk_tables()

def create_session(mode: str = "free_chat", scenario: str = "general") -> str:
    session_id = f"talk_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO talk_sessions (id, mode, scenario, started_at)
    VALUES (?, ?, ?, ?)
    """, (session_id, mode, scenario, time.time()))
    conn.commit()
    conn.close()
    return session_id

def add_turn(
    session_id: str,
    turn_idx: int,
    role: str,
    text_zh: str,
    text_pinyin: Optional[str] = None,
    text_translation: Optional[str] = None,
    audio_path: Optional[str] = None,
    pron_report: Optional[Dict[str, Any]] = None,
    corrections: Optional[List[Dict[str, Any]]] = None,
    latency_ms: Optional[float] = None
) -> int:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO talk_turns (
        session_id, turn_idx, role, text_zh, text_pinyin, text_translation,
        audio_path, pron_report_json, corrections_json, latency_ms, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        turn_idx,
        role,
        text_zh,
        text_pinyin or "",
        text_translation or "",
        audio_path or "",
        json.dumps(pron_report, ensure_ascii=False) if pron_report else "",
        json.dumps(corrections, ensure_ascii=False) if corrections else "",
        latency_ms or 0.0,
        time.time()
    ))
    turn_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return turn_id

def get_session_history(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT role, text_zh, text_pinyin, text_translation, pron_report_json, corrections_json
    FROM talk_turns
    WHERE session_id = ?
    ORDER BY turn_idx ASC
    LIMIT ?
    """, (session_id, limit))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for r in rows:
        history.append({
            "role": r["role"],
            "text_zh": r["text_zh"],
            "text_pinyin": r["text_pinyin"],
            "text_translation": r["text_translation"],
            "pron_report": json.loads(r["pron_report_json"]) if r["pron_report_json"] else None,
            "corrections": json.loads(r["corrections_json"]) if r["corrections_json"] else []
        })
    return history

def close_session(session_id: str, summary: Dict[str, Any]):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE talk_sessions
    SET ended_at = ?, summary_json = ?
    WHERE id = ?
    """, (time.time(), json.dumps(summary, ensure_ascii=False), session_id))
    conn.commit()
    conn.close()
