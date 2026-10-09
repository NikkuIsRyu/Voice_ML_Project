"""
Tests for Live AI Teacher ('Talk' Mode):
- WebSocket connection and session lifecycle
- Dynamic prompt builder with German explanations
- Sentence-level chunking for Chinese speech
- Talk session and turn persistence
"""

import sys
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tutor" / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from main import app
from talk.teacher_prompt import build_teacher_system_prompt
from talk.tts_client import split_sentences_zh
from talk.session_store import create_session, add_turn, get_session_history, close_session

class TestTalkPromptAndText:
    def test_german_prompt_builder(self):
        prompt = build_teacher_system_prompt(
            explain_lang="de",
            strictness="normal",
            scenario="restaurant",
            known_vocab=["你好", "谢谢"],
            due_words=["买"]
        )
        assert "Erklärungen auf DEUTSCH" in prompt
        assert "restaurant" in prompt.lower()
        assert "reply_zh" in prompt
        assert "reply_translation" in prompt
        assert "grammar_corrections" in prompt
        assert "pronunciation_focus" in prompt

    def test_sentence_splitting(self):
        text = "我今天也很忙。你吃了午饭没有？快去吃吧！"
        sentences = split_sentences_zh(text)
        assert len(sentences) == 3
        assert sentences[0] == "我今天也很忙。"
        assert sentences[1] == "你吃了午饭没有？"
        assert sentences[2] == "快去吃吧！"

class TestTalkSessionStore:
    def test_session_lifecycle(self):
        sid = create_session(mode="live_teacher", scenario="directions")
        assert sid.startswith("talk_")

        tid = add_turn(
            session_id=sid,
            turn_idx=1,
            role="user",
            text_zh="请问，地铁站在哪里？",
            latency_ms=120.0
        )
        assert tid > 0

        add_turn(
            session_id=sid,
            turn_idx=2,
            role="teacher",
            text_zh="地铁站就在前面，往前走两分钟就到了。你要去哪里？",
            text_translation="Die U-Bahn-Station ist gleich vorne, zwei Minuten geradeaus. Wohin möchtest du?",
            corrections=[]
        )

        history = get_session_history(sid, limit=5)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["text_zh"] == "请问，地铁站在哪里？"
        assert history[1]["role"] == "teacher"

        close_session(sid, {"turns": 2, "rating": "good"})

class TestTalkWebSocketIntegration:
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)

    def test_websocket_text_conversation(self, client):
        with client.websocket_connect("/ws/talk") as ws:
            # 1. Start session
            ws.send_json({
                "type": "start_session",
                "scenario": "free_chat",
                "strictness": "normal",
                "explain_lang": "de"
            })
            resp = ws.receive_json()
            assert resp["type"] == "session_started"
            assert "session_id" in resp

            # 2. Send text input
            ws.send_json({
                "type": "text_input",
                "text": "你好，今天天气真好！"
            })

            # 3. Receive user caption
            caption_msg = ws.receive_json()
            assert caption_msg["type"] == "caption_user"
            assert caption_msg["text"] == "你好，今天天气真好！"

            # 4. Receive reply start
            start_msg = ws.receive_json()
            assert start_msg["type"] == "reply_start"

            # 5. Receive reply text
            reply_msg = ws.receive_json()
            assert reply_msg["type"] == "reply_text"
            assert "reply_zh" in reply_msg
            assert "reply_translation" in reply_msg
            assert len(reply_msg["reply_zh"]) > 0
