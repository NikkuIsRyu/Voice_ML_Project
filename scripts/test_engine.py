import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "tutor/backend")

import soundfile as sf
from pronunciation_engine import get_pronunciation_engine

audio, sr = sf.read(r"output_v2/vocab/example_lesson/example_lesson_005_你好.wav")
engine = get_pronunciation_engine()
res = engine.evaluate(audio, sr, "你好")

print("Overall Score:", res["overall_score"])
print("Tone Accuracy:", res["tone_accuracy"])
for syl in res["syllables"]:
    ch = syl["char"]
    exp = syl["expected"]
    exp_t = syl["expected_tone"]
    hrd_t = syl["heard_tone"]
    iss = syl["issue"]
    t0 = syl["t_start"]
    t1 = syl["t_end"]
    print(f"{ch} ({exp}): expected T{exp_t}, heard T{hrd_t}, issue={iss}, span=[{t0}s - {t1}s]")
