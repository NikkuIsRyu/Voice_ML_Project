import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import settings

logger = logging.getLogger("talk.llm")

class TeacherLLMProvider:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self._client = None

        if self.provider == "gemini" and self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info(f"Gemini client initialized with model {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")

    def generate_reply(
        self,
        system_instruction: str,
        conversation_history: List[Dict[str, Any]],
        user_text: str,
        user_audio_bytes: Optional[bytes] = None,
        pron_report: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate structured teacher reply with German explanations and Chinese dialogue.
        Sends multimodal raw audio + local syllable measurement to Gemini.
        """
        if self._client and self.api_key:
            return self._call_gemini(
                system_instruction, conversation_history, user_text, user_audio_bytes, pron_report
            )
        else:
            return self._local_fallback_reply(user_text, pron_report)

    def _call_gemini(
        self,
        system_instruction: str,
        conversation_history: List[Dict[str, Any]],
        user_text: str,
        user_audio_bytes: Optional[bytes],
        pron_report: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        from google.genai import types

        # Build turn prompt with local evidence
        user_content_parts = []

        if user_audio_bytes and len(user_audio_bytes) > 0:
            user_content_parts.append(
                types.Part.from_bytes(
                    data=user_audio_bytes,
                    mime_type="audio/wav"
                )
            )

        pron_summary = ""
        if pron_report and "syllables" in pron_report:
            items = []
            for s in pron_report["syllables"]:
                st = f"'{s.get('char')}' ({s.get('expected')}): erwartet T{s.get('expected_tone')}, gehört T{s.get('heard_tone')}"
                if s.get("issue"):
                    st += f" [FEHLER: {s['issue']}]"
                items.append(st)
            pron_summary = "\nLokale akustische Ton- und Phonem-Messung:\n" + "\n".join(items)

        prompt_text = f"Der Schüler sagte: \"{user_text}\"\n{pron_summary}\n\nBitte antworte als Lehrerin im geforderten JSON-Format."
        user_content_parts.append(types.Part.from_text(text=prompt_text))

        # Format conversation history
        contents = []
        for h in conversation_history[-6:]:
            if h["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=h["text_zh"])]))
            elif h["role"] == "teacher":
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=h["text_zh"])]))

        contents.append(types.Content(role="user", parts=user_content_parts))

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.7
            )
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )

            text_resp = resp.text.strip()
            # Clean possible markdown fence
            if text_resp.startswith("```json"):
                text_resp = text_resp[7:]
            if text_resp.startswith("```"):
                text_resp = text_resp[3:]
            if text_resp.endswith("```"):
                text_resp = text_resp[:-3]
            return json.loads(text_resp.strip())
        except Exception as e:
            logger.error(f"Gemini API generation error: {e}")
            return self._local_fallback_reply(user_text, pron_report, error_note=str(e))

    def _local_fallback_reply(
        self,
        user_text: str,
        pron_report: Optional[Dict[str, Any]],
        error_note: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Graceful local rule-based response when GEMINI_API_KEY is not yet entered.
        Provides conversational continuity and highlights measured tone issues in German.
        """
        corrections = []
        pron_focus = []

        if pron_report and "syllables" in pron_report:
            for s in pron_report["syllables"]:
                if s.get("issue") == "tone":
                    pron_focus.append({
                        "syllable": s.get("expected", ""),
                        "note": f"Achte auf den Ton bei {s.get('char')}: erwartet Ton {s.get('expected_tone')}, gesprochen Ton {s.get('heard_tone')}."
                    })

        reply_zh = "你好！我很高兴和你用中文练习。你今天过得怎么样？"
        reply_pinyin = "nǐ hǎo! wǒ hěn gāoxìng hé nǐ yòng zhōngwén liànxí. nǐ jīntiān guò de zěnmeyàng?"
        reply_trans = "Hallo! Ich freue mich, mit dir Chinesisch zu üben. Wie war dein Tag heute?"

        if not self.api_key:
            corrections.append({
                "you_said": "Hinweis zum Gemini API Key",
                "better": "GEMINI_API_KEY in .env eintragen",
                "why": "Für dynamische KI-Antworten trage bitte deinen kostenlosen Google Gemini API-Schlüssel in die Datei .env ein."
            })

        return {
            "reply_zh": reply_zh,
            "reply_pinyin": reply_pinyin,
            "reply_translation": reply_trans,
            "intended_user_zh": user_text,
            "grammar_corrections": corrections,
            "pronunciation_focus": pron_focus,
            "drill": None,
            "new_words": []
        }

_llm_instance = None

def get_teacher_llm() -> TeacherLLMProvider:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = TeacherLLMProvider()
    return _llm_instance
