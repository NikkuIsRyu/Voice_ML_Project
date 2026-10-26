"""
backup_and_sync_google_drive.py
-------------------------------
1. Backs up live Anki database and automatic backups (local + Google Drive).
2. Archives existing .apkg deck on Google Drive into a backup folder.
3. Uploads newly generated 10-lesson .apkg decks to Google Drive.
4. Archives previous Google Drive audio into backup_folder to preserve quality progression.
5. Uploads complete Lessons 1-10 dialogues (standard + slow) and vocabularies (standard + slow).
6. Generates historical progression and audio catalog READMEs.
"""

import os
import sys
import shutil
import time
from pathlib import Path
from datetime import datetime

# Windows UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_BACKUPS_DIR = REPO_ROOT / "backups"
LOCAL_OUTPUT_V2 = REPO_ROOT / "output_v2"
LOCAL_OUTPUT_V2_SLOW = REPO_ROOT / "output_v2_slow"
LOCAL_DATA = REPO_ROOT / "data"
LOCAL_REPORTS = REPO_ROOT / "reports"

GDRIVE_ROOT = Path("G:/Meine Ablage/University")
GDRIVE_ANKI = GDRIVE_ROOT / "Anki-chinese"
GDRIVE_VOICE_V4 = GDRIVE_ROOT / "Chinese_Voice_ML_v4"

ANKI_USER_DIR = Path("C:/Users/nickb/AppData/Roaming/Anki2/Benutzer 1")

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def copy_tree_with_progress(src: Path, dst: Path, ext_filter=None):
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    total_bytes = 0
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            if ext_filter and not any(f.endswith(ext) for ext in ext_filter):
                continue
            src_file = Path(root) / f
            dst_file = target_dir / f
            shutil.copy2(src_file, dst_file)
            count += 1
            total_bytes += src_file.stat().st_size
    return count, total_bytes


def step1_backup_anki():
    print("\n" + "="*70)
    print("STEP 1: BACKING UP ANKI USER PROFILE & ARCHIVING PREVIOUS DECKS")
    print("="*70)

    # 1.1 Local backup of live Anki profile
    local_anki_backup = LOCAL_BACKUPS_DIR / f"anki_live_profile_backup_{TIMESTAMP}"
    local_anki_backup.mkdir(parents=True, exist_ok=True)

    copied_profile_files = []
    if ANKI_USER_DIR.exists():
        print(f"Reading live Anki profile from: {ANKI_USER_DIR}")
        for filename in ["collection.anki2", "collection.media.db2"]:
            fpath = ANKI_USER_DIR / filename
            if fpath.exists():
                shutil.copy2(fpath, local_anki_backup / filename)
                size_mb = fpath.stat().st_size / (1024 * 1024)
                print(f"  ✓ Backed up {filename} ({size_mb:.2f} MB)")
                copied_profile_files.append(filename)

        # Back up automatic .colpkg backups
        live_backups_dir = ANKI_USER_DIR / "backups"
        if live_backups_dir.exists():
            dest_colpkg_dir = local_anki_backup / "backups"
            dest_colpkg_dir.mkdir(parents=True, exist_ok=True)
            for colpkg in live_backups_dir.glob("*.colpkg"):
                shutil.copy2(colpkg, dest_colpkg_dir / colpkg.name)
                print(f"  ✓ Backed up colpkg: {colpkg.name}")

    print(f"Local Anki profile backup complete at: {local_anki_backup}")

    # 1.2 Copy Anki profile backup to Google Drive
    gdrive_anki_backup_dir = GDRIVE_ANKI / "backups" / f"anki_live_profile_backup_{TIMESTAMP}"
    gdrive_anki_backup_dir.mkdir(parents=True, exist_ok=True)
    count, total_b = copy_tree_with_progress(local_anki_backup, gdrive_anki_backup_dir)
    print(f"  ✓ Uploaded Anki profile backup to Google Drive ({count} files, {total_b / (1024*1024):.2f} MB): {gdrive_anki_backup_dir}")

    # 1.3 Archive existing Google Drive .apkg files
    gdrive_apkg_archive_dir = GDRIVE_ANKI / "backups" / "pre_v5_expansion_apkg"
    gdrive_apkg_archive_dir.mkdir(parents=True, exist_ok=True)
    for apkg in GDRIVE_ANKI.glob("*.apkg"):
        dest_archive = gdrive_apkg_archive_dir / apkg.name
        if not dest_archive.exists():
            shutil.copy2(apkg, dest_archive)
            print(f"  ✓ Archived existing Drive deck {apkg.name} to {gdrive_apkg_archive_dir}")

    # 1.4 Upload newly generated .apkg packages to Google Drive
    anki_out_dir = LOCAL_OUTPUT_V2 / "anki"
    if anki_out_dir.exists():
        print(f"\nUploading new .apkg decks from {anki_out_dir} to {GDRIVE_ANKI}...")
        for apkg in anki_out_dir.glob("*.apkg"):
            shutil.copy2(apkg, GDRIVE_ANKI / apkg.name)
            size_mb = apkg.stat().st_size / (1024 * 1024)
            print(f"  ✓ Uploaded to Drive: {apkg.name} ({size_mb:.2f} MB)")

    # 1.5 Write README in Anki backups
    anki_backup_readme = GDRIVE_ANKI / "backups" / "README.md"
    anki_backup_readme.write_text(
        f"# Anki Backups Directory\n\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## Subdirectories\n"
        f"- `anki_live_profile_backup_{TIMESTAMP}/`: Snapshot of active `collection.anki2` database and Anki `.colpkg` backups from `AppData/Roaming/Anki2/Benutzer 1`.\n"
        f"- `pre_v5_expansion_apkg/`: Historical `.apkg` packages prior to the Lessons 5-10 rewrite and vocabulary expansion.\n",
        encoding="utf-8"
    )
    print(f"  ✓ Created Anki backups README: {anki_backup_readme}")


def step2_archive_and_upload_audio():
    print("\n" + "="*70)
    print("STEP 2: ARCHIVING OLD AUDIO & UPLOADING NEW AUDIO TO GOOGLE DRIVE")
    print("="*70)

    gdrive_backup_folder = GDRIVE_VOICE_V4 / "backup_folder"
    gdrive_backup_folder.mkdir(parents=True, exist_ok=True)

    gdrive_new_output_v4 = GDRIVE_VOICE_V4 / "new_output_v4"

    # 2.1 Archive existing new_output_v4 on Drive if present
    if gdrive_new_output_v4.exists():
        archive_dest = gdrive_backup_folder / f"archive_v4_early_drafts_20261026_0722"
        if not archive_dest.exists():
            print(f"Moving existing Drive new_output_v4 to backup folder: {archive_dest}...")
            shutil.move(str(gdrive_new_output_v4), str(archive_dest))
            print(f"  ✓ Successfully archived previous new_output_v4 to: {archive_dest}")
        else:
            print(f"  Notice: Archive {archive_dest} already exists.")

    # 2.2 Recreate clean new_output_v4 for current complete release
    gdrive_new_output_v4.mkdir(parents=True, exist_ok=True)
    print(f"Created active audio release directory: {gdrive_new_output_v4}")

    # 2.3 Upload Dialogues 1.0x Standard
    print("\nUploading Dialogues 1.0x Standard (Lessons 2-10)...")
    src_dial_std = LOCAL_OUTPUT_V2 / "dialogues"
    dst_dial_std = gdrive_new_output_v4 / "dialogues_1.0x_standard"
    c, b = copy_tree_with_progress(src_dial_std, dst_dial_std)
    print(f"  ✓ Dialogues 1.0x: {c} files, {b / (1024*1024):.2f} MB transferred")

    # 2.4 Upload Dialogues 0.75x Slow
    print("\nUploading Dialogues 0.75x Slow (Lessons 2-10)...")
    src_dial_slow = LOCAL_OUTPUT_V2_SLOW / "dialogues"
    dst_dial_slow = gdrive_new_output_v4 / "dialogues_0.75x_slow"
    c, b = copy_tree_with_progress(src_dial_slow, dst_dial_slow)
    print(f"  ✓ Dialogues 0.75x: {c} files, {b / (1024*1024):.2f} MB transferred")

    # 2.5 Upload Vocab 1.0x Standard
    print("\nUploading Vocab 1.0x Standard (Lessons 1-10)...")
    src_vocab_std = LOCAL_OUTPUT_V2 / "vocab"
    dst_vocab_std = gdrive_new_output_v4 / "vocab_1.0x_standard"
    c, b = copy_tree_with_progress(src_vocab_std, dst_vocab_std)
    print(f"  ✓ Vocab 1.0x: {c} files, {b / (1024*1024):.2f} MB transferred")

    # 2.6 Upload Vocab 0.65x Slow
    print("\nUploading Vocab 0.65x Slow (Lessons 1-10)...")
    src_vocab_slow = LOCAL_OUTPUT_V2_SLOW / "vocab"
    dst_vocab_slow = gdrive_new_output_v4 / "vocab_0.65x_slow"
    c, b = copy_tree_with_progress(src_vocab_slow, dst_vocab_slow)
    print(f"  ✓ Vocab 0.65x: {c} files, {b / (1024*1024):.2f} MB transferred")

    # 2.7 Upload Transcripts / Texts
    print("\nUploading Transcripts & Vocab texts...")
    dst_texts_dial = gdrive_new_output_v4 / "dialogues_texts"
    c, b = copy_tree_with_progress(LOCAL_DATA / "dialogues", dst_texts_dial)
    print(f"  ✓ Dialogue Texts: {c} files transferred")

    dst_texts_vocab = gdrive_new_output_v4 / "vocab_texts"
    c, b = copy_tree_with_progress(LOCAL_DATA / "vocab", dst_texts_vocab)
    print(f"  ✓ Vocab Texts: {c} files transferred")

    # 2.8 Upload Reports
    print("\nUploading Reports...")
    dst_reports = gdrive_new_output_v4 / "reports"
    c, b = copy_tree_with_progress(LOCAL_REPORTS, dst_reports)
    print(f"  ✓ Reports: {c} files transferred")

    # 2.9 If AB comparison samples exist in the backup, copy them to new_output_v4 as well for easy access
    old_ab = gdrive_backup_folder / "archive_v4_early_drafts_20261026_0722" / "v4_ab_comparison_samples"
    if old_ab.exists():
        dst_ab = gdrive_new_output_v4 / "v4_ab_comparison_samples"
        c, b = copy_tree_with_progress(old_ab, dst_ab)
        print(f"  ✓ AB Comparison Samples preserved: {c} files transferred")

    # 2.10 Write Audio Catalog README
    catalog_path = gdrive_new_output_v4 / "README_AUDIO_CATALOG.md"
    catalog_path.write_text(
        f"# Chinese Voice ML v4 Audio Release Catalog\n\n"
        f"**Release Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## Overview\n"
        f"This directory contains the complete audio suite generated with fine-tuned GPT-SoVITS models "
        f"for textbook Lessons 1 through 10, including multi-speaker dialogues and dual-speed vocabulary.\n\n"
        f"## Directory Structure\n"
        f"- `dialogues_1.0x_standard/`: 18 composite dialogues (Lessons 2-10) at natural conversational speed (1.0x), plus individual line segments and JSON alignment timestamps.\n"
        f"- `dialogues_0.75x_slow/`: 18 composite dialogues (Lessons 2-10) at study speed (0.75x) for listening practice, with alignment timestamps.\n"
        f"- `vocab_1.0x_standard/`: 560 vocabulary terms across 10 lessons synthesized with `female_v4` (CSMSC 7.50h Studio model) at standard speed (1.0x).\n"
        f"- `vocab_0.65x_slow/`: 560 vocabulary terms across 10 lessons at slow speed (0.65x) for clear phoneme and tone distinction.\n"
        f"- `dialogues_texts/`: Cleaned text transcripts for all dialogues.\n"
        f"- `vocab_texts/`: Cleaned vocabulary word lists for all lessons.\n"
        f"- `reports/`: Generation reports, benchmark evaluation, and quality metrics.\n"
        f"- `v4_ab_comparison_samples/`: Side-by-side comparative WAV samples comparing previous v3 models against v4.\n\n"
        f"## Voice Profile Mapping\n"
        f"- **Female Role (Speaker A):** `female_v4` (Fine-tuned on 7.50h studio CSMSC recordings, CER 1.25%, natural pitch contour).\n"
        f"- **Male Role (Speaker B):** `male_v1` (Warm conversational male tone).\n"
        f"- **Third Role (Speaker C):** `male_zs` (Resonant male tone for multi-character scenarios).\n\n"
        f"## Historical Backups\n"
        f"All historical audio files and previous versions are preserved in `../backup_folder/`.\n",
        encoding="utf-8"
    )
    print(f"  ✓ Written catalog: {catalog_path}")

    # 2.11 Write Progression README in backup_folder
    progression_path = gdrive_backup_folder / "README_HISTORICAL_PROGRESSION.md"
    progression_path.write_text(
        f"# Chinese Voice ML - Historical Audio Progression & Evolution\n\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## Purpose\n"
        f"This folder preserves all older audio iterations to document both the **quality growth** (voice synthesis fidelity) "
        f"and the **knowledge quantity growth** (progression from basic lessons to full university curriculum).\n\n"
        f"## Evolution Timeline\n\n"
        f"### Stage 1: Legacy Audio (v1 - v3)\n"
        f"- **Folder:** `../legacy_dialogues_backup/`\n"
        f"- **Coverage:** Lessons 2 to 4 dialogues.\n"
        f"- **Model:** Early zero-shot reference cloning and base GPT-SoVITS checkpoint.\n"
        f"- **Characteristics:** Baseline intelligible audio, slight robotic artifacts on high tone transitions.\n\n"
        f"### Stage 2: v4 Early Session Drafts\n"
        f"- **Folder:** `archive_v4_early_drafts_20261026_0722/`\n"
        f"- **Coverage:** Lessons 2 to 7 dialogues.\n"
        f"- **Model:** CSMSC 7.50h fine-tuned `female_v4` checkpoint (Milestone D initial test).\n"
        f"- **Characteristics:** Massive leap in naturalness (CER 1.25%), studio warmth.\n\n"
        f"### Stage 3: Current Complete Release (Lessons 1 - 10)\n"
        f"- **Folder:** `../new_output_v4/`\n"
        f"- **Coverage:** Full curriculum Lessons 1 to 10 (18 dialogues + 560 vocabulary terms).\n"
        f"- **Features:**\n"
        f"  - Dual dialogue speeds: 1.0x standard conversational + 0.75x study speed.\n"
        f"  - Dual vocabulary speeds: 1.0x standard + 0.65x slow tonal study speed.\n"
        f"  - Word-level timestamp JSON alignments for all dialogues.\n"
        f"  - Multi-speaker casting (Speaker A: female_v4, Speaker B: male_v1, Speaker C: male_zs).\n"
        f"  - Master Anki deck + 10 individual lesson packages in `../Anki-chinese/`.\n",
        encoding="utf-8"
    )
    print(f"  ✓ Written progression guide: {progression_path}")


def main():
    start_time = time.time()
    print("Starting Backup and Google Drive Sync...")
    step1_backup_anki()
    step2_archive_and_upload_audio()
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"ALL OPERATIONS COMPLETED SUCCESSFULLY IN {elapsed:.1f}s")
    print("="*70)


if __name__ == "__main__":
    main()
