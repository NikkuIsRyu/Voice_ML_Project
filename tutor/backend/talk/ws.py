import os
import io
import time
import json
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import soundfile as sf
import numpy as np

from config import settings
from audio_utils import load_and_resample
from scoring import PronunciationScorer
from pronunciation_engine import get_pronunciation_engine
from database import get_due_srs_words, get_db

from .session_store import create_session, add_turn, get_session_history, close_session
from .teacher_prompt import build_teacher_system_prompt
from .llm import get_teacher_llm
from .tts_client import get_tts_client, split_sentences_zh

logger = logging.getLogger("talk.ws")

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TALK_UPLOADS = PROJECT_ROOT / "tutor" / "uploads" / "talk"
TALK_UPLOADS.mkdir(parents=True, exist_ok=True)

# Shared ASR Scorer singleton
_scorer = None
def get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = PronunciationScorer()
    return _scorer

@router.websocket("/ws/talk")
async def talk_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    session_id = None
    turn_idx = 0
    scenario = "free_chat"
    strictness = settings.CORRECTION_STRICTNESS
    explain_lang = settings.EXPLAIN_LANGUAGE
    audio_buffer = bytearray()
    interrupted = False

    llm = get_teacher_llm()
    tts = get_tts_client()
    scorer = get_scorer()
    pron_engine = get_pronunciation_engine()

    # Load known vocab for context
    known_vocab = []
    due_words = []
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT text FROM srs_items LIMIT 50")
        known_vocab = [r[0] for r in c.fetchall()]
        conn.close()
        due_items = get_due_srs_words()
        due_words = [d.get("text", "") for d in due_items if d.get("text")]
    except Exception as e:
        logger.warning(f"Could not load known vocab: {e}")

    try:
        while True:
            # Handle incoming messages (binary audio chunk or JSON control)
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                audio_buffer.extend(message["bytes"])
                continue

            if "text" in message and message["text"]:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "start_session":
                    scenario = data.get("scenario", "free_chat")
                    strictness = data.get("strictness", strictness)
                    explain_lang = data.get("explain_lang", explain_lang)
                    session_id = create_session(mode="live_teacher", scenario=scenario)
                    turn_idx = 0
                    audio_buffer.clear()

                    await websocket.send_json({
                        "type": "session_started",
                        "session_id": session_id,
                        "scenario": scenario,
                        "explain_lang": explain_lang,
                        "strictness": strictness
                    })
                    continue

                elif msg_type == "audio_chunk":
                    raw_b64 = data.get("data", "")
                    if raw_b64:
                        audio_buffer.extend(base64.b64decode(raw_b64))
                    continue

                elif msg_type == "interrupt":
                    interrupted = True
                    audio_buffer.clear()
                    await websocket.send_json({"type": "interrupted"})
                    continue

                elif msg_type == "end_of_speech" or msg_type == "text_input":
                    if not session_id:
                        session_id = create_session(mode="live_teacher", scenario=scenario)

                    turn_idx += 1
                    interrupted = False
                    start_turn_time = time.time()
                    user_text = ""
                    user_audio_bytes = None
                    wav_path = None
                    pron_res = None

                    if msg_type == "text_input":
                        user_text = data.get("text", "").strip()
                        if not user_text:
                            continue
                        await websocket.send_json({"type": "caption_user", "text": user_text})
                    else:
                        # Process recorded speech audio
                        if len(audio_buffer) < 400:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Zu kurze Audioaufnahme. Bitte sprich etwas länger."
                            })
                            audio_buffer.clear()
                            continue

                        # Save raw audio chunk
                        raw_audio_copy = bytes(audio_buffer)
                        audio_buffer.clear()

                        try:
                            user_audio, sr = load_and_resample(raw_audio_copy)
                        except Exception as e:
                            await websocket.send_json({"type": "error", "message": f"Audiodekodierung fehlgeschlagen: {e}"})
                            continue

                        # Save 16kHz WAV for ASR and persistence
                        wav_filename = f"{session_id}_turn_{turn_idx}.wav"
                        wav_path = TALK_UPLOADS / wav_filename
                        sf.write(str(wav_path), user_audio, sr)
                        with open(wav_path, "rb") as f:
                            user_audio_bytes = f.read()

                        # Step 1: Transcribe via Whisper on GPU
                        t_asr_start = time.time()
                        try:
                            user_text = scorer.transcribe(str(wav_path))
                        except Exception as e:
                            logger.error(f"ASR error: {e}")
                            user_text = ""

                        if not user_text:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Es konnte keine chinesische Sprache erkannt werden. Bitte versuche es erneut."
                            })
                            continue

                        # Send user caption immediately
                        await websocket.send_json({"type": "caption_user", "text": user_text})

                        # Step 2: Acoustic tone & phoneme evaluation (A4 engine)
                        try:
                            pron_res = pron_engine.evaluate(user_audio, sr, user_text)
                        except Exception as e:
                            logger.warning(f"Pronunciation engine warning: {e}")
                            pron_res = None

                    # Step 3: LLM generation
                    await websocket.send_json({"type": "reply_start"})

                    sys_prompt = build_teacher_system_prompt(
                        explain_lang=explain_lang,
                        strictness=strictness,
                        scenario=scenario,
                        known_vocab=known_vocab,
                        due_words=due_words
                    )
                    history = get_session_history(session_id, limit=8)

                    llm_start = time.time()
                    llm_reply = llm.generate_reply(
                        system_instruction=sys_prompt,
                        conversation_history=history,
                        user_text=user_text,
                        user_audio_bytes=user_audio_bytes,
                        pron_report=pron_res
                    )
                    llm_duration = time.time() - llm_start

                    # Broadcast text reply and explanations
                    await websocket.send_json({
                        "type": "reply_text",
                        "reply_zh": llm_reply.get("reply_zh", ""),
                        "reply_pinyin": llm_reply.get("reply_pinyin", ""),
                        "reply_translation": llm_reply.get("reply_translation", "")
                    })

                    if llm_reply.get("grammar_corrections") or llm_reply.get("pronunciation_focus"):
                        await websocket.send_json({
                            "type": "corrections",
                            "grammar_corrections": llm_reply.get("grammar_corrections", []),
                            "pronunciation_focus": llm_reply.get("pronunciation_focus", [])
                        })

                    # Step 4: Sentence-level TTS audio streaming
                    reply_zh = llm_reply.get("reply_zh", "")
                    sentences = split_sentences_zh(reply_zh)

                    for seq_idx, sent in enumerate(sentences):
                        if interrupted:
                            break
                        wav_data = await tts.synthesize_sentence(sent, speed_factor=1.0)
                        if wav_data:
                            b64_audio = base64.b64encode(wav_data).decode("utf-8")
                            await websocket.send_json({
                                "type": "reply_audio",
                                "sentence": sent,
                                "audio_base64": b64_audio,
                                "seq": seq_idx,
                                "total": len(sentences)
                            })

                    # Step 5: Send per-turn pronunciation report if audio was evaluated
                    if pron_res:
                        await websocket.send_json({
                            "type": "pron_report",
                            "target_text": user_text,
                            "overall_score": pron_res.get("overall_score", 0),
                            "tone_accuracy": pron_res.get("tone_accuracy", 0),
                            "char_accuracy": pron_res.get("char_accuracy", 0),
                            "syllables": pron_res.get("syllables", [])
                        })

                    total_latency_ms = (time.time() - start_turn_time) * 1000.0

                    # Step 6: Persist turns to database
                    add_turn(
                        session_id=session_id,
                        turn_idx=turn_idx * 2 - 1,
                        role="user",
                        text_zh=user_text,
                        audio_path=str(wav_path) if wav_path else None,
                        pron_report=pron_res,
                        latency_ms=total_latency_ms
                    )

                    add_turn(
                        session_id=session_id,
                        turn_idx=turn_idx * 2,
                        role="teacher",
                        text_zh=llm_reply.get("reply_zh", ""),
                        text_pinyin=llm_reply.get("reply_pinyin", ""),
                        text_translation=llm_reply.get("reply_translation", ""),
                        corrections=llm_reply.get("grammar_corrections", [])
                    )

                    await websocket.send_json({"type": "turn_complete", "turn_idx": turn_idx})

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from talk session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket session exception: {e}")
        try:
            await websocket.send_json({"type": "error", "message": f"Server-Fehler: {e}"})
        except Exception:
            pass
