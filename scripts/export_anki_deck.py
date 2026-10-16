#!/usr/bin/env python3
"""
Export Chinese Vocabulary Anki Decks with Refined GPT-SoVITS / CSMSC Audio.

This script parses Chinese vocabulary files and packages them into Anki (.apkg) decks
complete with:
  - High-fidelity refined audio clips (CSMSC / Biaobei v2ProPlus)
  - Color-coded Pinyin (Pleco standard tone colors)
  - Ruby annotations (Pinyin above Hanzi)
  - Recognition & Listening comprehension card types
  - Subdecks structured per lesson under a unified master deck
  - Standalone per-lesson .apkg exports

Usage:
    python scripts/export_anki_deck.py
    python scripts/export_anki_deck.py --cards-mode both --export-individual
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import genanki
from pypinyin import pinyin, Style

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# === Tone Colors & Rules ===
TONE_MAP = {
    1: set('āēīōūǖĀĒĪŌŪǕ'),
    2: set('áéíóúǘÁÉÍÓÚǗ'),
    3: set('ǎěǐǒǔǚǍĚǏǑǓǙ'),
    4: set('àèìòùǜÀÈÌÒÙǛ')
}

def detect_tone(s: str) -> int:
    """Detect Mandarin tone (1-5) from tone-marked pinyin syllable."""
    for tone, chars in TONE_MAP.items():
        if any(c in chars for c in s):
            return tone
    return 5


def slice_raw_pinyin(hz: str, raw_py: str) -> List[Tuple[str, str, int]]:
    """
    Align Hanzi characters with raw pinyin syllables using pypinyin length guidance.
    Preserves text-specific tone sandhi and neutral tone marks from textbook.
    Returns: list of (character, syllable_str, tone_int)
    """
    # Special cases: erhua words pronounced as a single syllable
    if hz in ("哪儿", "这儿", "那儿"):
        return [(hz, raw_py, detect_tone(raw_py))]

    clean_hz = [c for c in hz if '\u4e00' <= c <= '\u9fff']
    if not clean_hz:
        return [(hz, raw_py, detect_tone(raw_py))]

    py_guide = [s[0] for s in pinyin("".join(clean_hz), style=Style.TONE)]

    tokens = raw_py.split()
    if len(tokens) == len(clean_hz):
        return [(c, t, detect_tone(t)) for c, t in zip(clean_hz, tokens)]

    # Normalized character index mapping (excluding apostrophes and spaces)
    norm_chars = []
    idx_map = []
    for i, ch in enumerate(raw_py):
        if ch not in " '·,-":
            norm_chars.append(ch)
            idx_map.append(i)

    slices = []
    curr = 0
    success = True
    for g in py_guide:
        L = len(g)
        if curr + L <= len(norm_chars):
            start_idx = idx_map[curr]
            end_idx = idx_map[curr + L - 1] + 1
            segment = raw_py[start_idx:end_idx]
            slices.append(segment)
            curr += L
        else:
            success = False
            break

    if success and curr == len(norm_chars) and len(slices) == len(clean_hz):
        return [(c, s, detect_tone(s)) for c, s in zip(clean_hz, slices)]

    # Fallback to pypinyin guide
    return [(c, s, detect_tone(s)) for c, s in zip(clean_hz, py_guide)]


def build_colored_pinyin_html(aligned: List[Tuple[str, str, int]]) -> str:
    """Build colored pinyin HTML string with tone spans."""
    spans = []
    for _, syl, tone in aligned:
        spans.append(f'<span class="tone tone-{tone}">{syl}</span>')
    return " ".join(spans)


def build_ruby_html(aligned: List[Tuple[str, str, int]]) -> str:
    """Build HTML ruby elements (Hanzi with Pinyin above)."""
    rubies = []
    for char, syl, tone in aligned:
        rubies.append(f'<ruby>{char}<rt class="tone tone-{tone}">{syl}</rt></ruby>')
    return "".join(rubies)


# === CSS Styling ===
CARD_CSS = """
/* Anki Card Styling for Refined Chinese Vocab */
.card {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 16px;
    text-align: center;
    color: #2c3e50;
    background-color: #f8fafc;
    line-height: 1.5;
    padding: 20px;
    margin: 0;
}

.nightMode.card, .night_mode .card {
    background-color: #0f172a;
    color: #f1f5f9;
}

.vocab-container {
    max-width: 540px;
    margin: 0 auto;
    background: #ffffff;
    border-radius: 16px;
    padding: 32px 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
    border: 1px solid #e2e8f0;
}

.nightMode .vocab-container, .night_mode .vocab-container {
    background: #1e293b;
    border-color: #334155;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}

/* Lesson Badge */
.lesson-badge {
    display: inline-block;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #6366f1;
    background: #eef2ff;
    border-radius: 9999px;
    margin-bottom: 20px;
}

.nightMode .lesson-badge, .night_mode .lesson-badge {
    color: #a5b4fc;
    background: #312e81;
}

/* Hanzi Typography */
.hanzi-main {
    font-size: 52px;
    font-weight: 700;
    line-height: 1.25;
    margin: 16px 0;
    color: #0f172a;
    font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans CN", sans-serif;
}

.nightMode .hanzi-main, .night_mode .hanzi-main {
    color: #f8fafc;
}

/* Ruby Typography */
ruby {
    font-size: 48px;
    font-weight: 700;
    margin: 0 2px;
    ruby-position: over;
    font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
}

rt {
    font-size: 16px;
    font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    letter-spacing: 0.5px;
    padding-bottom: 2px;
}

/* Pinyin Display */
.pinyin-display {
    font-size: 26px;
    font-weight: 600;
    letter-spacing: 1px;
    margin: 14px 0 6px 0;
}

/* Tone Colors (Pleco Standard) */
.tone-1 { color: #e11d48 !important; } /* High Level (Red) */
.tone-2 { color: #16a34a !important; } /* Rising (Green) */
.tone-3 { color: #2563eb !important; } /* Dipping (Blue) */
.tone-4 { color: #9333ea !important; } /* Falling (Purple) */
.tone-5 { color: #64748b !important; } /* Neutral (Slate Gray) */

.nightMode .tone-1, .night_mode .tone-1 { color: #fb7185 !important; }
.nightMode .tone-2, .night_mode .tone-2 { color: #4ade80 !important; }
.nightMode .tone-3, .night_mode .tone-3 { color: #60a5fa !important; }
.nightMode .tone-4, .night_mode .tone-4 { color: #c084fc !important; }
.nightMode .tone-5, .night_mode .tone-5 { color: #94a3b8 !important; }

/* German Meaning */
.meaning-display {
    font-size: 20px;
    font-weight: 500;
    color: #334155;
    margin: 16px 0 8px 0;
}

.nightMode .meaning-display, .night_mode .meaning-display {
    color: #cbd5e1;
}

/* Divider */
hr.soft-divider {
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 20px 0;
}

.nightMode hr.soft-divider, .night_mode hr.soft-divider {
    background: #334155;
}

/* Audio & Listening Prompt */
.audio-prompt {
    font-size: 40px;
    margin: 18px 0;
    color: #4f46e5;
}

.nightMode .audio-prompt, .night_mode .audio-prompt {
    color: #818cf8;
}

.card-hint {
    font-size: 14px;
    color: #64748b;
    margin-top: 8px;
}

.nightMode .card-hint, .night_mode .card-hint {
    color: #94a3b8;
}

/* Tone Legend */
.tone-legend {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px dashed #e2e8f0;
    font-size: 11px;
    font-weight: 600;
}

.nightMode .tone-legend, .night_mode .tone-legend {
    border-color: #334155;
}

.tone-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.tone-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.dot-1 { background-color: #e11d48; }
.dot-2 { background-color: #16a34a; }
.dot-3 { background-color: #2563eb; }
.dot-4 { background-color: #9333ea; }
.dot-5 { background-color: #64748b; }
"""

# === Anki Card Templates ===
CARD1_FRONT = """
<div class="vocab-container">
    <div class="lesson-badge">{{Lesson}} • {{LessonTopic}}</div>
    <div class="hanzi-main">{{Hanzi}}</div>
    <div class="card-hint">Wie lauten Aussprache (Pinyin) und deutsche Bedeutung?</div>
</div>
"""

CARD1_BACK = """
<div class="vocab-container">
    <div class="lesson-badge">{{Lesson}} • {{LessonTopic}}</div>
    <div class="ruby-display">{{HanziRuby}}</div>
    <div class="pinyin-display">{{PinyinColored}}</div>
    
    <hr class="soft-divider">
    <div class="meaning-display">{{German}}</div>
    
    <div style="margin-top: 16px;">
        {{Audio}}
    </div>

    <div class="tone-legend">
        <span class="tone-pill"><span class="tone-dot dot-1"></span> T1</span>
        <span class="tone-pill"><span class="tone-dot dot-2"></span> T2</span>
        <span class="tone-pill"><span class="tone-dot dot-3"></span> T3</span>
        <span class="tone-pill"><span class="tone-dot dot-4"></span> T4</span>
        <span class="tone-pill"><span class="tone-dot dot-5"></span> Neutral</span>
    </div>
</div>
"""

CARD2_FRONT = """
<div class="vocab-container">
    <div class="lesson-badge">{{Lesson}} • {{LessonTopic}}</div>
    <div class="audio-prompt">🎧</div>
    <div style="margin: 12px 0;">
        {{Audio}}
    </div>
    <div class="card-hint">Hören & Verstehen: Welches Wort wird gesprochen?</div>
    {{#German}}
    <div style="margin-top: 12px; font-size: 13px;">
        {{hint:German}}
    </div>
    {{/German}}
</div>
"""

CARD2_BACK = """
<div class="vocab-container">
    <div class="lesson-badge">{{Lesson}} • {{LessonTopic}}</div>
    <div style="margin-bottom: 12px;">
        {{Audio}}
    </div>
    <div class="ruby-display">{{HanziRuby}}</div>
    <div class="pinyin-display">{{PinyinColored}}</div>
    
    <hr class="soft-divider">
    <div class="meaning-display">{{German}}</div>

    <div class="tone-legend">
        <span class="tone-pill"><span class="tone-dot dot-1"></span> T1</span>
        <span class="tone-pill"><span class="tone-dot dot-2"></span> T2</span>
        <span class="tone-pill"><span class="tone-dot dot-3"></span> T3</span>
        <span class="tone-pill"><span class="tone-dot dot-4"></span> T4</span>
        <span class="tone-pill"><span class="tone-dot dot-5"></span> Neutral</span>
    </div>
</div>
"""


def get_deterministic_id(key: str) -> int:
    """Generate a stable 31-bit positive integer ID from a string key."""
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:7]
    return int(h, 16)


def create_anki_model(cards_mode: str = "both") -> genanki.Model:
    """Create genanki Model with custom fields, CSS, and card templates."""
    fields = [
        {"name": "Hanzi"},
        {"name": "HanziRuby"},
        {"name": "Pinyin"},
        {"name": "PinyinColored"},
        {"name": "German"},
        {"name": "Audio"},
        {"name": "Lesson"},
        {"name": "LessonTopic"},
        {"name": "ItemNumber"},
    ]

    templates = []
    if cards_mode in ("both", "recognition"):
        templates.append({
            "name": "1. Lesen & Erkennen (Hanzi ➔ Audio)",
            "qfmt": CARD1_FRONT,
            "afmt": CARD1_BACK,
        })
    if cards_mode in ("both", "listening"):
        templates.append({
            "name": "2. Hören & Verstehen (Audio ➔ Hanzi)",
            "qfmt": CARD2_FRONT,
            "afmt": CARD2_BACK,
        })

    model_id = get_deterministic_id(f"chinese_voice_ml_model_{cards_mode}")
    return genanki.Model(
        model_id=model_id,
        name="Chinesisch Vokabeln (Refined Voice Model)",
        fields=fields,
        templates=templates,
        css=CARD_CSS,
    )


def parse_lesson_file(filepath: Path) -> Tuple[str, str, List[dict]]:
    """
    Parse a vocab lesson text file.
    Returns: (lesson_code, lesson_topic, entries)
    """
    lines = filepath.read_text(encoding="utf-8").splitlines()
    lesson_code = filepath.stem
    lesson_topic = lesson_code

    # Extract topic header if available
    if "example" in lesson_code.lower():
        lesson_topic = "Grundwortschatz"
    else:
        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                header_text = line.lstrip("#").strip()
                # e.g. "Lektion 1: Begrüßung und Kennenlernen"
                match = re.match(r"(?:Lektion\s*\d+\s*:\s*)?(.*)", header_text, re.IGNORECASE)
                if match and match.group(1):
                    lesson_topic = match.group(1).strip()
                break

    entries = []
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            hanzi, raw_pinyin, german = parts[0], parts[1], parts[2]
            entries.append({
                "line_idx": line_idx,
                "hanzi": hanzi,
                "raw_pinyin": raw_pinyin,
                "german": german
            })

    return lesson_code, lesson_topic, entries


def format_lesson_label(lesson_code: str) -> str:
    """Format lesson code to friendly label like 'Lektion 01'."""
    m = re.match(r"lesson(\d+)", lesson_code, re.IGNORECASE)
    if m:
        return f"Lektion {int(m.group(1)):02d}"
    if "example" in lesson_code.lower():
        return "Beispiel-Lektion"
    return lesson_code


def export_decks(vocab_dir: Path, audio_dir: Path, output_dir: Path,
                 master_deck_name: str = "Chinesisch Sprachkurs (Refined Voice)",
                 cards_mode: str = "both", export_individual: bool = True):
    """Generate master and individual Anki .apkg files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_anki_model(cards_mode=cards_mode)

    txt_files = sorted(vocab_dir.glob("*.txt"))
    if not txt_files:
        print(f"Error: No .txt files found in {vocab_dir}")
        return

    all_media_files = []
    subdecks = []
    master_notes_count = 0
    lesson_stats = []

    for txt_file in txt_files:
        lesson_code, lesson_topic, entries = parse_lesson_file(txt_file)
        lesson_audio_dir = audio_dir / lesson_code
        lesson_label = format_lesson_label(lesson_code)

        # Full hierarchical deck name
        full_subdeck_name = f"{master_deck_name}::{lesson_label} - {lesson_topic}"
        subdeck_id = get_deterministic_id(full_subdeck_name)
        subdeck = genanki.Deck(deck_id=subdeck_id, name=full_subdeck_name)

        lesson_media_files = []
        matched_entries = 0

        for item_num, entry in enumerate(entries, start=1):
            line_idx = entry["line_idx"]
            hanzi = entry["hanzi"]
            raw_pinyin = entry["raw_pinyin"]
            german = entry["german"]

            # Locate WAV file
            # Format: <lesson_code>_<line_idx:03d>_<hanzi>.wav
            wav_filename = f"{lesson_code}_{line_idx:03d}_{hanzi}.wav"
            wav_path = lesson_audio_dir / wav_filename

            if not wav_path.exists():
                # Try finding any file matching line_idx
                candidates = list(lesson_audio_dir.glob(f"{lesson_code}_{line_idx:03d}_*.wav"))
                if candidates:
                    wav_path = candidates[0]
                    wav_filename = wav_path.name

            if not wav_path.exists():
                print(f"  WARNING: Missing audio file for {lesson_code} line {line_idx} ({hanzi}): {wav_path}")
                audio_tag = ""
            else:
                audio_tag = f"[sound:{wav_filename}]"
                lesson_media_files.append(str(wav_path.resolve()))
                all_media_files.append(str(wav_path.resolve()))
                matched_entries += 1

            aligned = slice_raw_pinyin(hanzi, raw_pinyin)
            colored_pinyin = build_colored_pinyin_html(aligned)
            ruby_html = build_ruby_html(aligned)

            # Deterministic note GUID
            note_guid = genanki.guid_for(lesson_code, str(line_idx), hanzi)

            note = genanki.Note(
                model=model,
                fields=[
                    hanzi,
                    ruby_html,
                    raw_pinyin,
                    colored_pinyin,
                    german,
                    audio_tag,
                    lesson_label,
                    lesson_topic,
                    f"{item_num:03d}",
                ],
                guid=note_guid,
                tags=[f"Lektion_{lesson_code}", "Voice_ML", "Biaobei_v2ProPlus"]
            )
            subdeck.add_note(note)

        subdecks.append(subdeck)
        master_notes_count += len(entries)
        lesson_stats.append({
            "lesson": lesson_label,
            "topic": lesson_topic,
            "items": len(entries),
            "audio_matched": matched_entries,
            "deck": subdeck,
            "media": lesson_media_files,
            "code": lesson_code
        })

    # 1. Export Master Package with all subdecks
    master_package_path = output_dir / "Chinesisch_Vokabeln_Alle_Lektionen.apkg"
    print(f"\nBuilding Master Anki Deck: {master_package_path.name}...")
    # Unique media files list
    unique_media = sorted(list(set(all_media_files)))
    master_pkg = genanki.Package(subdecks)
    master_pkg.media_files = unique_media
    master_pkg.write_to_file(str(master_package_path))
    file_size_mb = os.path.getsize(master_package_path) / (1024 * 1024)
    print(f"  ✓ Master deck written: {master_package_path} ({file_size_mb:.2f} MB)")

    # 2. Export individual lesson decks if requested
    if export_individual:
        print("\nExporting individual lesson decks...")
        umlaut_map = {
            'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
            'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
            'ß': 'ss'
        }
        for stat in lesson_stats:
            clean_str = stat["topic"]
            for k, v in umlaut_map.items():
                clean_str = clean_str.replace(k, v)
            safe_topic = re.sub(r"[^\w\s-]", "", clean_str).strip().replace(" ", "_")
            out_file = output_dir / f"{stat['lesson'].replace(' ', '_')}_{safe_topic}.apkg"
            ind_deck = genanki.Deck(
                deck_id=get_deterministic_id(f"ind_{stat['code']}"),
                name=f"{master_deck_name}::{stat['lesson']} - {stat['topic']}"
            )
            # Transfer notes
            for note in stat["deck"].notes:
                ind_deck.add_note(note)

            ind_pkg = genanki.Package(ind_deck)
            ind_pkg.media_files = stat["media"]
            ind_pkg.write_to_file(str(out_file))
            ind_size_mb = os.path.getsize(out_file) / (1024 * 1024)
            print(f"  ✓ {out_file.name} ({stat['items']} notes, {len(stat['media'])} audio, {ind_size_mb:.2f} MB)")

    print("\n" + "=" * 65)
    print("ANKI DECK GENERATION SUMMARY")
    print("=" * 65)
    print(f"Output Directory:    {output_dir.resolve()}")
    print(f"Total Lessons:       {len(lesson_stats)}")
    print(f"Total Notes:         {master_notes_count}")
    cards_per_note = 2 if cards_mode == "both" else 1
    print(f"Total Flashcards:    {master_notes_count * cards_per_note} ({cards_mode} mode)")
    print(f"Total Audio Files:   {len(unique_media)}")
    print(f"Master Package:      {master_package_path.name} ({file_size_mb:.2f} MB)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Export Chinese Vocabulary Anki Decks with Refined Voice Audio",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--vocab-dir",
        default="data/vocab",
        help="Directory containing vocab .txt files (default: data/vocab)"
    )
    parser.add_argument(
        "--audio-dir",
        default="output_v2/vocab",
        help="Directory containing refined audio .wav files (default: output_v2/vocab)"
    )
    parser.add_argument(
        "--output-dir",
        default="output_v2/anki",
        help="Directory to save .apkg decks (default: output_v2/anki)"
    )
    parser.add_argument(
        "--deck-name",
        default="Chinesisch Sprachkurs (Refined Voice)",
        help="Master deck name in Anki"
    )
    parser.add_argument(
        "--cards-mode",
        choices=["both", "recognition", "listening"],
        default="both",
        help="Card types to generate: 'both' (default), 'recognition', or 'listening'"
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Disable generating individual per-lesson .apkg files"
    )

    args = parser.parse_args()
    export_decks(
        vocab_dir=Path(args.vocab_dir),
        audio_dir=Path(args.audio_dir),
        output_dir=Path(args.output_dir),
        master_deck_name=args.deck_name,
        cards_mode=args.cards_mode,
        export_individual=not args.no_individual
    )


if __name__ == "__main__":
    main()
