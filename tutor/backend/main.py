import os
import sys
import json
import time
import glob
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Ensure backend directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Local imports
from database import get_db, init_db, log_practice, get_stats
from audio_utils import load_and_resample, extract_f0_contour, align_and_score_pitch
from scoring import PronunciationScorer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VOCAB_DIR = PROJECT_ROOT / "data" / "vocab"
DIALOGUE_DIR = PROJECT_ROOT / "data" / "dialogues"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_V2_DIR = PROJECT_ROOT / "output_v2"
UPLOADS_DIR = PROJECT_ROOT / "tutor" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Antigravity Chinese Pronunciation Tutor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scorer = None

def get_scorer():
    global scorer
    if scorer is None:
        scorer = PronunciationScorer()
    return scorer

def populate_initial_srs():
    """Populate SRS table with all available vocab words from lesson files."""
    conn = get_db()
    cursor = conn.cursor()
    count = 0
    for lesson_path in sorted(VOCAB_DIR.glob("*.txt")):
        lesson_name = lesson_path.stem
        with open(lesson_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    chinese = parts[0]
                    pinyin_text = parts[1]
                    trans = parts[2]
                    item_id = f"{lesson_name}_{chinese}"
                    cursor.execute("""
                    INSERT OR IGNORE INTO srs_items (item_id, text, pinyin, translation, lesson)
                    VALUES (?, ?, ?, ?, ?)
                    """, (item_id, chinese, pinyin_text, trans, lesson_name))
                    count += 1
    conn.commit()
    conn.close()

# Populate DB on start
populate_initial_srs()

import threading
def _warmup_worker():
    try:
        sc = get_scorer()
        sc._get_model()
        print("PronunciationScorer warmed up and ready in memory.")
    except Exception as e:
        print(f"Scorer warmup notice: {e}")

threading.Thread(target=_warmup_worker, daemon=True).start()

# ----------------- APIs -----------------

@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "Chinese Pronunciation Tutor", "time": time.time()}

@app.get("/api/stats")
def stats_endpoint():
    return get_stats()

@app.get("/api/lessons")
def list_lessons():
    """List all vocabulary lessons."""
    lessons = []
    for p in sorted(VOCAB_DIR.glob("*.txt")):
        name = p.stem
        # Count words
        words = 0
        with open(p, "r", encoding="utf-8") as f:
            words = sum(1 for line in f if line.strip() and not line.startswith("#"))
        lessons.append({
            "id": name,
            "title": name.replace("_", " ").capitalize(),
            "word_count": words
        })
    return {"lessons": lessons}

def find_audio_file(category: str, subfolder: str, filename: str) -> Optional[Path]:
    """Helper to locate an audio file in v2 or v1 output directories."""
    candidates = [
        OUTPUT_V2_DIR / category / subfolder / filename,
        OUTPUT_V2_DIR / category / filename,
        OUTPUT_DIR / category / subfolder / filename,
        OUTPUT_DIR / category / filename,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    # Fallback recursive search
    for root in [OUTPUT_V2_DIR / category, OUTPUT_DIR / category]:
        matches = list(root.rglob(filename))
        if matches:
            return matches[0]
    return None

@app.get("/api/lessons/{lesson_id}")
def get_lesson_words(lesson_id: str):
    """Get all words in a lesson with native audio paths."""
    lesson_path = VOCAB_DIR / f"{lesson_id}.txt"
    if not lesson_path.exists():
        raise HTTPException(status_code=404, detail="Lesson not found")

    words = []
    with open(lesson_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                chinese = parts[0]
                pinyin_text = parts[1]
                trans = parts[2]
                
                # Check for audio file (exact or with index prefix)
                cand = list((OUTPUT_V2_DIR / "vocab" / lesson_id).glob(f"*{chinese}*.wav"))
                if not cand:
                    cand = list((OUTPUT_DIR / "vocab" / lesson_id).glob(f"*{chinese}*.wav"))
                if not cand:
                    cand = list(OUTPUT_V2_DIR.glob(f"*{chinese}*.wav"))
                if not cand:
                    cand = list(OUTPUT_DIR.glob(f"*{chinese}*.wav"))

                audio_file = cand[0] if cand else None
                audio_filename = audio_file.name if audio_file else f"{lesson_id}_{chinese}.wav"

                words.append({
                    "id": f"{lesson_id}_{chinese}",
                    "chinese": chinese,
                    "pinyin": pinyin_text,
                    "translation": trans,
                    "audio_filename": audio_filename,
                    "audio_url": f"/api/audio/vocab/{lesson_id}/{audio_filename}" if audio_file else None
                })
    return {"lesson_id": lesson_id, "words": words}

@app.get("/api/dialogues")
def list_dialogues():
    """List all dialogue scenarios."""
    dialogues = []
    for p in sorted(DIALOGUE_DIR.glob("*.txt")):
        name = p.stem
        lines_count = 0
        roles = set()
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lines_count += 1
                if ":" in line:
                    roles.add(line.split(":")[0].strip())
                elif "：" in line:
                    roles.add(line.split("：")[0].strip())
        dialogues.append({
            "id": name,
            "title": name.replace("_", " ").capitalize(),
            "lines_count": lines_count,
            "roles": sorted(list(roles))
        })
    return {"dialogues": dialogues}

@app.get("/api/dialogues/{dialogue_id}")
def get_dialogue_lines(dialogue_id: str):
    """Get all lines in a dialogue."""
    dialogue_path = DIALOGUE_DIR / f"{dialogue_id}.txt"
    if not dialogue_path.exists():
        raise HTTPException(status_code=404, detail="Dialogue not found")

    # Match audio files sequentially
    dialogue_files_v2 = sorted(list((OUTPUT_V2_DIR / "dialogues" / dialogue_id).glob("*.wav")))
    dialogue_files_v1 = sorted(list((OUTPUT_DIR / "dialogues" / dialogue_id).glob("*.wav")))
    dialogue_files = dialogue_files_v2 if dialogue_files_v2 else dialogue_files_v1

    lines = []
    with open(dialogue_path, "r", encoding="utf-8") as f:
        idx = 1
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            speaker = "Speaker"
            text = line
            if ":" in line:
                speaker, text = line.split(":", 1)
            elif "：" in line:
                speaker, text = line.split("：", 1)
            speaker = speaker.strip()
            text = text.strip()

            audio_file = dialogue_files[idx - 1] if (0 <= idx - 1 < len(dialogue_files)) else None
            audio_filename = audio_file.name if audio_file else f"{dialogue_id}_line{idx:02d}.wav"

            lines.append({
                "line_index": idx,
                "speaker": speaker,
                "text": text,
                "audio_filename": audio_filename,
                "audio_url": f"/api/audio/dialogues/{dialogue_id}/{audio_filename}" if audio_file else None
            })
            idx += 1
    return {"dialogue_id": dialogue_id, "lines": lines}

@app.get("/api/audio/{category}/{filepath:path}")
def stream_audio(category: str, filepath: str):
    """Stream audio file (prioritizes v2, falls back to v1, supports subdirectories)."""
    p2 = OUTPUT_V2_DIR / category / filepath
    if p2.exists() and p2.is_file():
        return FileResponse(p2, media_type="audio/wav")
    p1 = OUTPUT_DIR / category / filepath
    if p1.exists() and p1.is_file():
        return FileResponse(p1, media_type="audio/wav")

    # Search by filename in category folder
    fname = Path(filepath).name
    for root in [OUTPUT_V2_DIR / category, OUTPUT_DIR / category]:
        if root.exists():
            matches = list(root.rglob(fname))
            if matches:
                return FileResponse(matches[0], media_type="audio/wav")

    raise HTTPException(status_code=404, detail="Audio file not found")

@app.post("/api/score/vocab")
def score_vocab_practice(
    audio: UploadFile = File(...),
    item_id: str = Form(...),
    target_text: str = Form(...),
    target_pinyin: str = Form(""),
    audio_filename: str = Form("")
):
    """
    Score user pronunciation of a vocabulary word:
    - Extracts F0 pitch curves for native audio and user audio
    - Aligns and scores pitch similarity via DTW
    - Runs ASR & phoneme/tone diagnostic evaluation
    - Updates SRS database
    """
    # 1. Save user recording
    temp_wav = UPLOADS_DIR / f"rec_{int(time.time()*1000)}.wav"
    content = audio.file.read()
    with open(temp_wav, "wb") as f:
        f.write(content)

    # 2. Resample user audio
    try:
        user_audio, sr = load_and_resample(temp_wav)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decoding error: {e}")

    # 3. Pronunciation & Tone Scoring
    sc = get_scorer()
    eval_res = sc.score_audio(str(temp_wav), target_text)

    # 4. Pitch (F0) Extraction for User Audio
    user_f0_info = extract_f0_contour(user_audio, sr=sr, n_points=80)

    # 5. Native reference audio F0 extraction
    ref_f0_info = {"norm_contour": [0.0]*80, "has_voicing": False}
    pitch_score = 75.0  # default reasonable fallback
    aligned_ref = [0.0]*80
    aligned_usr = [0.0]*80

    ref_path = None
    if audio_filename or target_text:
        fname = Path(audio_filename).name if audio_filename else ""
        for root in [OUTPUT_V2_DIR / "vocab", OUTPUT_DIR / "vocab"]:
            if root.exists():
                if fname:
                    matches = list(root.rglob(fname))
                    if matches:
                        ref_path = matches[0]
                        break
                if not ref_path and target_text:
                    matches = list(root.rglob(f"*{target_text}*.wav"))
                    if matches:
                        ref_path = matches[0]
                        break

    if ref_path and ref_path.exists():
        try:
            ref_audio, _ = load_and_resample(ref_path)
            ref_f0_info = extract_f0_contour(ref_audio, sr=sr, n_points=80)
            if ref_f0_info["has_voicing"] and user_f0_info["has_voicing"]:
                pitch_score, aligned_ref, aligned_usr = align_and_score_pitch(
                    ref_f0_info["norm_contour"],
                    user_f0_info["norm_contour"]
                )
        except Exception as e:
            print(f"Ref pitch extraction warning: {e}")

    # 6. Overall Composite Score (40% Phoneme/Char, 40% Tone, 20% Pitch trajectory)
    char_acc = eval_res["char_accuracy"]
    tone_acc = eval_res["tone_accuracy"]
    overall_score = round(char_acc * 0.40 + tone_acc * 0.40 + pitch_score * 0.20, 1)

    # 7. Log to database
    log_practice(
        item_type="vocab",
        item_id=item_id,
        target_text=target_text,
        target_pinyin=target_pinyin,
        recognized_text=eval_res["recognized_text"],
        cer=eval_res["cer"],
        tone_accuracy=tone_acc,
        pitch_score=pitch_score,
        overall_score=overall_score,
        error_details=eval_res["diagnostics"],
        audio_filename=temp_wav.name
    )

    return {
        "overall_score": overall_score,
        "char_accuracy": char_acc,
        "tone_accuracy": tone_acc,
        "pitch_score": pitch_score,
        "target_text": target_text,
        "recognized_text": eval_res["recognized_text"],
        "diagnostics": eval_res["diagnostics"],
        "pitch_curves": {
            "ref": ref_f0_info["norm_contour"],
            "user": user_f0_info["norm_contour"],
            "aligned_ref": aligned_ref,
            "aligned_user": aligned_usr
        }
    }

# Mount static frontend
FRONTEND_DIR = PROJECT_ROOT / "tutor" / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
