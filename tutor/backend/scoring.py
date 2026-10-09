import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from pypinyin import pinyin, Style
import numpy as np
import torch

from sandhi import apply_tone_sandhi

# Windows DLL helper for faster-whisper CTranslate2
try:
    torch_lib = r"D:\miniconda3\envs\GPTSoVits\Lib\site-packages\torch\lib"
    if os.path.isdir(torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
        os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")
    from faster_whisper import WhisperModel
except Exception as e:
    WhisperModel = None

# Global GPU lock to serialize heavy inference passes
GPU_LOCK = threading.Lock()

TONE_DESCRIPTIONS = {
    1: "Tone 1 (High Level, 55): Keep your pitch steady, even, and high — like singing a sustained note.",
    2: "Tone 2 (Rising, 35): Start mid-level and rise smoothly upward — like asking 'What?'.",
    3: "Tone 3 (Low Dipping, 214): Dip low into your chest voice before a light rise — express thoughtful hesitation.",
    4: "Tone 4 (Falling, 51): Start high and drop sharply and decisively — like an emphatic command.",
    5: "Tone 5 (Neutral): Short, light, and relaxed with no sustained pitch."
}

INITIAL_ADVICE = {
    "zh": "Retroflex 'zh': Curl the tip of your tongue back against the hard palate (behind the alveolar ridge).",
    "ch": "Aspirated retroflex 'ch': Tongue curled back like 'zh', but release a strong puff of air.",
    "sh": "Retroflex 'sh': Tongue curled back, let air hiss softly over the tongue blade.",
    "r": "Retroflex 'r': Tongue curled back with voiced vibration, similar to the 's' in 'pleasure'.",
    "z": "Flat sibilant 'z': Tongue tip flat against the back of your front teeth (no curl).",
    "c": "Flat sibilant 'c': Tongue tip against front teeth, release a strong burst of air like 'ts'.",
    "s": "Flat sibilant 's': Tongue tip against front teeth, hiss like English 's'."
}

class PronunciationScorer:
    def __init__(
        self,
        asr_model_path: str = "D:/.D Dokuments/Uni Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_models/asr/faster-whisper-large-v3",
        device: Optional[str] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.compute_type = "int8_float16" if self.device == "cuda" else "int8"
        self.model_path = asr_model_path
        self._model = None

    def _get_model(self):
        if self._model is None:
            if WhisperModel is None:
                raise RuntimeError("faster_whisper is not available")
            print(f"Loading Whisper model from {self.model_path} ({self.device}, {self.compute_type})...")
            self._model = WhisperModel(self.model_path, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, wav_path: str) -> str:
        """Transcribe speech in audio to Chinese text."""
        model = self._get_model()
        with GPU_LOCK:
            segments, _ = model.transcribe(wav_path, language="zh", beam_size=5)
            segments = list(segments)
        return "".join(s.text for s in segments).strip()

    def analyze_phonemes(self, text: str) -> List[Dict[str, Any]]:
        """
        Decompose text into character, pinyin, initial, final, and tone number.
        """
        clean_text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
        res = []
        if not clean_text:
            return res

        pyn_tones = pinyin(clean_text, style=Style.TONE3, neutral_tone_with_five=True)
        pyn_marks = pinyin(clean_text, style=Style.TONE)
        initials = pinyin(clean_text, style=Style.INITIALS, strict=False)
        finals = pinyin(clean_text, style=Style.FINALS_TONE3, neutral_tone_with_five=True)

        for i, char in enumerate(clean_text):
            raw_pyn = pyn_tones[i][0] if i < len(pyn_tones) else ""
            tone_num = 5
            for digit in "12345":
                if digit in raw_pyn:
                    tone_num = int(digit)
                    break

            init = initials[i][0] if i < len(initials) else ""
            fin = finals[i][0] if i < len(finals) else ""
            # Strip digit from final
            fin_clean = re.sub(r"\d", "", fin)
            mark = pyn_marks[i][0] if i < len(pyn_marks) else ""

            res.append({
                "char": char,
                "pinyin": mark,
                "tone": tone_num,
                "initial": init,
                "final": fin_clean
            })
        return res

    def score_audio(self, wav_path: str, target_text: str) -> Dict[str, Any]:
        """
        Transcribe audio, evaluate segmental and tonal match, and generate diagnostic feedback.
        """
        model = self._get_model()
        with GPU_LOCK:
            segments, _ = model.transcribe(wav_path, language="zh", beam_size=5)
            segments = list(segments)
        recognized = "".join(s.text for s in segments).strip()

        target_analysis = apply_tone_sandhi(self.analyze_phonemes(target_text))
        rec_analysis = self.analyze_phonemes(recognized)

        target_chars = [x["char"] for x in target_analysis]
        rec_chars = [x["char"] for x in rec_analysis]

        # Calculate character match
        char_matches = 0
        tone_matches = 0
        diagnostics = []

        for i, tgt in enumerate(target_analysis):
            matched = False
            rec_match = None
            if i < len(rec_analysis):
                rec_item = rec_analysis[i]
                if rec_item["char"] == tgt["char"]:
                    matched = True
                    rec_match = rec_item
                elif rec_item["initial"] == tgt["initial"] and rec_item["final"] == tgt["final"]:
                    # Phoneme match but different character/tone
                    rec_match = rec_item

            expected_tone = tgt.get("expected_tone", tgt["tone"])
            if matched:
                char_matches += 1
                tone_matches += 1
                comment = "Accurate pronunciation and tone."
                if tgt.get("sandhi_applied"):
                    comment = f"Accurate pronunciation with correct tone sandhi (Tone {expected_tone})."
                diagnostics.append({
                    "char": tgt["char"],
                    "target_pinyin": tgt["pinyin"],
                    "target_tone": expected_tone,
                    "status": "correct",
                    "comment": comment
                })
            elif rec_match and rec_match["tone"] != expected_tone:
                # Tone mistake
                diagnostics.append({
                    "char": tgt["char"],
                    "target_pinyin": tgt["pinyin"],
                    "target_tone": expected_tone,
                    "detected_tone": rec_match["tone"],
                    "status": "tone_error",
                    "comment": f"Tone mismatch! Detected Tone {rec_match['tone']}, expected Tone {expected_tone}. {TONE_DESCRIPTIONS.get(expected_tone, '')}"
                })
            else:
                # Segmental / phone error
                advice = ""
                if tgt["initial"] in INITIAL_ADVICE:
                    advice = f" Note: {INITIAL_ADVICE[tgt['initial']]}"
                diagnostics.append({
                    "char": tgt["char"],
                    "target_pinyin": tgt["pinyin"],
                    "target_tone": expected_tone,
                    "status": "pronunciation_error",
                    "comment": f"Character was mispronounced or skipped.{advice}"
                })

        total_chars = max(len(target_chars), 1)
        cer = max(0.0, 1.0 - (char_matches / total_chars))
        tone_accuracy = round((tone_matches / total_chars) * 100.0, 1)
        char_accuracy = round((char_matches / total_chars) * 100.0, 1)

        return {
            "target_text": target_text,
            "recognized_text": recognized,
            "char_accuracy": char_accuracy,
            "tone_accuracy": tone_accuracy,
            "cer": round(cer, 3),
            "diagnostics": diagnostics
        }
