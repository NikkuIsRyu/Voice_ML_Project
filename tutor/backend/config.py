import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

class Settings:
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    EXPLAIN_LANGUAGE: str = os.getenv("EXPLAIN_LANGUAGE", "de").lower()  # 'de', 'en', 'zh'
    CORRECTION_STRICTNESS: str = os.getenv("CORRECTION_STRICTNESS", "normal").lower()
    TTS_URL: str = os.getenv("TTS_URL", "http://127.0.0.1:9880").strip()
    TUTOR_PORT: int = int(os.getenv("TUTOR_PORT", "8000"))

settings = Settings()
