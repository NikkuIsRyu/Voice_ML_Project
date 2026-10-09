import os
import re
import io
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import httpx

from config import settings

logger = logging.getLogger("talk.tts")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TTS_CACHE_DIR = PROJECT_ROOT / "tutor" / "uploads" / "tts_cache"
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REF_AUDIO = str(PROJECT_ROOT / "data" / "train_v2" / "csmsc" / "csmsc_ref.wav")
DEFAULT_REF_TEXT = "宝马配挂跛骡鞍，貂蝉怨枕，董翁榻。"

def split_sentences_zh(text: str) -> List[str]:
    """Splits Chinese text into natural sentence fragments at punctuation boundaries."""
    # Split on 。！？； and English equivalents
    raw_parts = re.split(r'([。！？；!?\n]+)', text)
    sentences = []
    current = ""
    for p in raw_parts:
        if not p:
            continue
        if re.match(r'[。！？；!?\n]+', p):
            current += p.strip()
            if current:
                sentences.append(current)
                current = ""
        else:
            current += p.strip()
    if current:
        sentences.append(current)
    return [s.strip() for s in sentences if s.strip()]

class TTSClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.TTS_URL).rstrip("/")
        self.ref_audio_path = DEFAULT_REF_AUDIO
        self.ref_prompt_text = DEFAULT_REF_TEXT

    async def is_available(self) -> bool:
        """Check if api_v2 server on 9880 is reachable."""
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                resp = await client.get(f"{self.base_url}/")
                return resp.status_code in (200, 404, 405)
        except Exception:
            return False

    async def synthesize_sentence(
        self,
        text: str,
        speed_factor: float = 1.0
    ) -> Optional[bytes]:
        """
        Synthesizes a single sentence through api_v2 with disk caching.
        Returns raw WAV bytes.
        """
        if not text.strip():
            return None

        # Check local disk cache
        cache_key = hashlib.md5(f"{text.strip()}_{speed_factor:.2f}".encode("utf-8")).hexdigest()
        cache_file = TTS_CACHE_DIR / f"{cache_key}.wav"
        if cache_file.exists():
            return cache_file.read_bytes()

        payload = {
            "text": text.strip(),
            "text_lang": "zh",
            "ref_audio_path": self.ref_audio_path,
            "prompt_text": self.ref_prompt_text,
            "prompt_lang": "zh",
            "speed_factor": speed_factor,
            "streaming_mode": False,
            "media_type": "wav"
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(f"{self.base_url}/tts", json=payload)
                if resp.status_code == 200 and len(resp.content) > 100:
                    wav_bytes = resp.content
                    try:
                        cache_file.write_bytes(wav_bytes)
                    except Exception as e:
                        logger.warning(f"Failed to cache TTS audio: {e}")
                    return wav_bytes
                else:
                    logger.error(f"TTS synthesis failed ({resp.status_code}): {resp.text[:120]}")
                    return None
        except Exception as e:
            logger.error(f"TTS server connection error: {e}")
            return None

_tts_instance = None

def get_tts_client() -> TTSClient:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSClient()
    return _tts_instance
