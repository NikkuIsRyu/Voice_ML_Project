#!/usr/bin/env python3
"""
Ingest prepared Chinese learning materials from Google Drive (Chinese-Materials-prepared).
Extracts:
1. Dialogues for Lessons 5, 6, 7, 8, 9, 10 into data/dialogues/DialogueX_Y.txt.
2. Vocabularies for Lessons 7, 8, 9, 10 into data/vocab/lessonXX.txt.
3. Updates configs/dialogue_speakers.yaml with voice mappings for all dialogues.
"""

import os
import re
import sys
import shutil
from pathlib import Path
import yaml
from pypinyin import pinyin, Style

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path("G:/Meine Ablage/University/Chinese-Materials-prepared")
DIALOGUES_DIR = PROJECT_ROOT / "data" / "dialogues"
VOCAB_DIR = PROJECT_ROOT / "data" / "vocab"
BACKUP_DIR = PROJECT_ROOT / "backups" / "pre_v5_drive_ingest"

def backup_existing():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dia_backup = BACKUP_DIR / "dialogues"
    dia_backup.mkdir(parents=True, exist_ok=True)
    for f in DIALOGUES_DIR.glob("Dialogue*.txt"):
        shutil.copy2(f, dia_backup / f.name)
    print(f"Backed up existing dialogues to {dia_backup}")


def extract_dialogues():
    print("\n--- EXTRACTING DIALOGUES (Lessons 5-10) ---")
    dialogue_map = {}
    for les in range(5, 11):
        md_file = SOURCE_DIR / f"nikku_lektion_{les:02d}_umgeschrieben.md"
        if not md_file.exists():
            print(f"Warning: {md_file} not found!")
            continue
        content = md_file.read_text(encoding="utf-8")
        # Find all dialogues: ### Dialog X.Y: ...
        dia_sections = list(re.finditer(r"###\s+Dialog\s+([\d\.]+).*?\r?\n(.*?)(?=\r?\n###|\r?\n##|\Z)", content, re.DOTALL))
        for match in dia_sections:
            dia_num = match.group(1).strip()
            dia_body = match.group(2)
            # Match table rows: | Turn | Sprecher | Hanzi | Pinyin | Deutsche Übersetzung |
            rows = re.findall(r"\|\s*\*\*?(\d+)\*\*?\s*\|\s*\*\*?([A-Za-z]+)\*\*?\s*\|\s*([^\|]+)\|\s*([^\|]+)\|", dia_body)
            dia_id = f"Dialogue{dia_num.replace('.', '_')}"
            turns = []
            for t_idx, spk, hz, py in rows:
                hz_clean = hz.strip()
                turns.append(f"{spk.strip()}: {hz_clean}")
            dialogue_map[dia_id] = turns
            out_file = DIALOGUES_DIR / f"{dia_id}.txt"
            out_file.write_text("\n".join(turns) + "\n", encoding="utf-8")
            print(f"  -> Wrote {dia_id}.txt ({len(turns)} turns) to {out_file}")

    return dialogue_map


def update_speaker_config(dialogue_ids):
    cfg_path = PROJECT_ROOT / "configs" / "dialogue_speakers.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for dia_id in dialogue_ids:
        if dia_id not in cfg["dialogues"]:
            cfg["dialogues"][dia_id] = {
                "A": "female_v4",
                "B": "male_v1",
                "C": "male_zs"
            }
        else:
            # Update to female_v4 if female_v3 was used
            if cfg["dialogues"][dia_id].get("A") == "female_v3":
                cfg["dialogues"][dia_id]["A"] = "female_v4"

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"\nUpdated {cfg_path} with {len(dialogue_ids)} dialogues.")


def extract_vocabularies():
    print("\n--- EXTRACTING VOCABULARIES ---")
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    
    for les in range(7, 11):
        f_orig = SOURCE_DIR / f"lektion_{les:02d}_original_backup.md"
        f_nikk = SOURCE_DIR / f"nikku_lektion_{les:02d}_umgeschrieben.md"
        
        # 1. Parse orig backup
        orig_entries = []
        if f_orig.exists():
            t_orig = f_orig.read_text(encoding="utf-8")
            m_orig = re.search(r"##\s+.*?Neue Wörter.*?\r?\n(.*?)(?=\r?\n##|\Z)", t_orig, re.DOTALL)
            if m_orig:
                rows = re.findall(r"\|\s*\*\*?([^\*\|]+?)\*\*?\s*\|\s*([^\|]+)\|\s*([^\|]+)\|", m_orig.group(1))
                for r in rows:
                    if not re.match(r"^(Pinyin|Hanzi|:---)", r[0].strip()):
                        raw_py = r[0].strip()
                        raw_hz = r[1].strip()
                        raw_de = r[2].strip()
                        orig_entries.append((raw_hz, raw_py, raw_de))

        # 2. Parse nikku rewritten core vocab
        nikk_entries = []
        if f_nikk.exists():
            t_nikk = f_nikk.read_text(encoding="utf-8")
            m_nikk = re.search(r"###\s+A\.\s+Kernvokabular.*?\r?\n(.*?)(?=\r?\n###|\r?\n##|\Z)", t_nikk, re.DOTALL)
            if m_nikk:
                rows = re.findall(r"\|\s*\*\*?([^\*\|]+?)\*\*?\s*\|\s*([^\|]+)\|\s*([^\|]+)\|\s*([^\|]+)\|", m_nikk.group(1))
                for r in rows:
                    if not re.match(r"^(Hanzi|:---)", r[0].strip()):
                        hz = r[0].strip()
                        py = r[1].strip()
                        de = r[3].strip()
                        nikk_entries.append((hz, py, de))

        print(f"Lesson {les:02d}: {len(orig_entries)} orig entries, {len(nikk_entries)} nikku entries")
        
        seen_hz = set()
        compiled_lines = []
        
        def clean_hz_text(h: str) -> str:
            # Strip outer punctuation, ellipses, plus notes
            h = re.sub(r"\+.*$", "", h) # e.g. + Nachname
            h = re.sub(r"[…\.\?\!？!。，,（\(\)）]", "", h)
            h = h.strip()
            return h

        def add_item(hz_item, py_item, de_item):
            # Check for splitting symbols ↔, /, oder
            delims_pattern = r"\s*(?:↔|/)\s*"
            
            # If multiple items separated by slash or arrow
            if re.search(delims_pattern, hz_item):
                hz_parts = re.split(delims_pattern, hz_item)
                py_parts = re.split(delims_pattern, py_item)
                de_parts = re.split(delims_pattern, de_item)
                
                for idx, hp in enumerate(hz_parts):
                    pp = py_parts[idx] if idx < len(py_parts) else ""
                    dp = de_parts[idx] if idx < len(de_parts) else de_item
                    add_item(hp, pp, dp)
                return

            # Handle parenthetical variants like 桌（子） -> 桌子
            # or 特色（菜） -> 特色菜 and 特色
            paren_match = re.search(r"（([^）]+)）|\(([^\)]+)\)", hz_item)
            if paren_match:
                inside = paren_match.group(1) or paren_match.group(2)
                base = re.sub(r"（[^）]+）|\([^\)]+\)", "", hz_item).strip()
                full = re.sub(r"（[^）]+）|\([^\)]+\)", inside, hz_item).strip()
                if full:
                    add_item(full, py_item, de_item)
                if base and base != full:
                    add_item(base, py_item, de_item)
                return

            hz_clean = clean_hz_text(hz_item)

            # Skip grammar frames with spaces or ellipses that leave incomplete pieces
            pure_cn = re.sub(r"[^\u4e00-\u9fff]", "", hz_clean)
            if not pure_cn or len(pure_cn) != len(hz_clean):
                # If there are still non-chinese chars like spaces, just keep pure_cn if pure_cn >= 1
                if not pure_cn:
                    return
                hz_clean = pure_cn
                
            if hz_clean in seen_hz:
                return

            # Clean pinyin
            clean_py = py_item.strip()
            clean_py = re.sub(r"[…\(\)（）]", "", clean_py).strip()
            clean_py = re.sub(r"^\.\.\.\s*", "", clean_py).strip()
            clean_py = re.sub(r"\s*[\?\!？！]$", "", clean_py).strip()
            
            # If pinyin empty or not matching character count roughly, generate via pypinyin
            if not clean_py or clean_py == "…" or len(clean_py.split()) != len(pure_cn):
                py_guide = pinyin(pure_cn, style=Style.TONE)
                clean_py = " ".join([item[0] for item in py_guide])

            # Clean german explanation
            de_clean = re.sub(r"[\*\_]", "", de_item).strip()
            seen_hz.add(hz_clean)
            compiled_lines.append(f"{hz_clean} | {clean_py} | {de_clean}")

        # Add all orig entries first
        for hz, py, de in orig_entries:
            add_item(hz, py, de)

        # Add any nikku core vocab words not yet covered
        for hz, py, de in nikk_entries:
            add_item(hz, py, de)

        topic_names = {
            7: "Wegbeschreibung, Verkehrsmittel & Orientierung",
            8: "Beim Einkauf, Elektronikzubehör & Kleidung",
            9: "Arbeitszeiten, Verabredung & Freizeit",
            10: "Restaurantbesuch, Bestellen & Bezahlen"
        }
        topic = topic_names.get(les, f"Lektion {les:02d}")
        header = f"# Lektion {les}: {topic}\n"
        out_txt = VOCAB_DIR / f"lesson{les:02d}.txt"
        out_txt.write_text(header + "\n".join(compiled_lines) + "\n", encoding="utf-8")
        print(f"  -> Generated {out_txt.name} with {len(compiled_lines)} vocabulary items.")


if __name__ == "__main__":
    backup_existing()
    dialogue_map = extract_dialogues()
    update_speaker_config(list(dialogue_map.keys()))
    extract_vocabularies()
    print("\nIngestion complete!")
