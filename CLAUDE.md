# Voice_ML_Project — Architecture, Handoff & Claude Operational Guide

> **Agent Guardrail & Token Conservation Rule:**
> You are the **Architect & Reviewer**. Gemini Flash acts as the sequential implementer.
> **DO NOT** recursively crawl directories (`ls -R`, `dir /s`), and **DO NOT** read large raw data files (`*.json` > 50 lines, `*.wav`, `*.npz`, `*.pth`, `*.colpkg`).
> Everything you need to understand project state, history, and architecture is summarized in this file (~1.3k tokens). If you need to inspect a specific function, request or read only that single file.

---

## 1. System & Runtime Environment
- **OS & Hardware:** Windows 11, Intel CPU, NVIDIA RTX 4060 Ti (8 GB VRAM).
- **Python Environment:** `D:\miniconda3\envs\GPTSoVits\python.exe`
- **VRAM Budget:** Strictly $\le 7.0\text{ GB}$. Whisper large-v3 (`float16`, CUDA) and TTS API must not train/run heavy jobs concurrently. DataLoader `num_workers=0`.
- **Git & Remote:** `git@github-voice:NikkuIsRyu/Voice_ML_Project.git` via dedicated SSH key (`~/.ssh/id_ed25519_voice_ml`).
- **Unified Launcher:** [`start_all.bat`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/start_all.bat) starts TTS Engine (Port 9880) and Tutor Web Server (Port 8000).

---

## 2. Progress & Architecture Across Milestones

### Milestone A: Pronunciation Engine & Live AI Teacher (Conv: `55e65b52`)
- **Fast Pitch Tracking:** Replaced slow pYIN with `librosa.yin` + RMS gating in [`tutor/backend/audio_utils.py`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/tutor/backend/audio_utils.py) ($12\text{ s} \rightarrow 0.02\text{ s}$).
- **GPU ASR Intelligibility:** Faster-Whisper `large-v3` on CUDA ($0.48\text{ s}$ latency).
- **Acoustic Tone Classifier & Sandhi:**
  - 21-dim MLP classifier trained on 20,908 CSMSC syllables ($87.1\%$ accuracy on T1–T4).
  - Rule-based tone sandhi ([`tutor/backend/sandhi.py`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/tutor/backend/sandhi.py)): 3-3 sandhi, "一" (yī) and "不" (bù) inflection rules.
- **WebSocket Live AI Teacher:** [`tutor/backend/talk/ws.py`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/tutor/backend/talk/ws.py) with pedagogical German explanations, scenario roleplay, and frontend mic-orb UI (`talk.js`).

### Milestone B: Git Baseline & Remote Synchronization (Conv: `362d4368`)
- Configured `.gitignore` to strictly exclude models (`*.pth`, `*.ckpt`), audio (`*.wav`, `output/`), datasets, and Anki packages.
- All code synchronized with GitHub `main` branch with clean tags.

### Milestone C: Voice-ML Upgrade v3 Pipeline (Conv: `1b3b27e1`)
Executed Phase 0 through Phase 7 strictly following Claude's architecture plan:
- **`v3-vad` (Phase 6):** Pre-normalization Butterworth bandpass filter ($70\text{--}5000\text{ Hz}$) and speech trimming (`trim_speech`), plus relative percentile RMS gating in F0 extraction.
- **`v3-female` (Phase 1):** CSMSC corpus expanded to 2.5 h, fine-tuned GPT-SoVITS v2 for 10 epochs with curated reference audio.
- **`v3-male` (Phase 2):** AISHELL-3 speaker SS0009 extracted (2.5 h) to train `male_v1`; configured `male_zs` zero-shot profile for 3-speaker dialogues.
- **`v3-dialogues` (Phase 3):** [`scripts/generate_multi_speaker_dialogues.py`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/scripts/generate_multi_speaker_dialogues.py) synthesizes single-file dialogues with natural room-tone pauses, speaker switches, and millisecond JSON timestamps.
- **`v3-examples` (Phase 5):** 345 contextual example sentences generated ([`data/vocab/vocab_example_sentences.json`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/data/vocab/vocab_example_sentences.json)) and synthesized.
- **`v3-anki-dual` (Phase 4):** [`scripts/export_anki_deck.py`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/scripts/export_anki_deck.py) embeds dual audio (1.0x standard + 0.65x slow) and example sentences. Uses `--preserve-model` with deterministic `note_guid` to preserve 100% of existing user learning progress; auto-generates `.colpkg` backups.
- **`v3-complete` (Phase 7):** Release tag created. All 18 unit tests passing.

### Milestone D: Voice-ML Upgrade v4 Voices & Dialogues (Conv: `d943cf12`)
- **`v4-dialogues-text`:** Rewrote Lessons 2–4 dialogues (0% textbook overlap) + added Lessons 5–7 (108 turns total).
- **`v4-female`:** Expanded CSMSC corpus to 7.50h (7,081 clips), trained SoVITS (10 ep, 11,810 steps) & GPT (12 ep) with curated reference prompt `csmsc_ref.wav`. Whisper large-v3 CER: 5.70% (0 outliers).
- **`v4-male` & `v4-male-b`:** Data inventory showed < 20 min unused corpus for SSB0817; existing high-quality `male_v1` and zero-shot `male_zs` preserved to protect voice quality.
- **`v4-dialogues-audio`:** Synthesized all 12 dialogues (Lessons 2–7) in standard 1.0x and slow 0.75x with ms alignment timestamps. Activated `female_v4` in `configs/dialogue_speakers.yaml`.
- **`v4-complete`:** All 50 unit tests passing. Anki deck untouched per user directive.

### Milestone E: Drive Prepared Materials & Curriculum Expansion (Lessons 5–10) (Conv: `deedd389`)
- **Ingested Prepared Materials:** Processed Google Drive materials from `Chinese-Materials-prepared` (`nikku_lektion_05_umgeschrieben.md` through `nikku_lektion_10_umgeschrieben.md`).
- **Curriculum Expansion (Lessons 8–10):** Added Dialogues 8.1, 8.2, 9.1, 9.2, 10.1, 10.2 and synchronized official rewritten texts for Dialogues 5.1–7.2 (88 turns total across 12 dialogues).
- **Multi-Speaker Audio Generation:** Synthesized all 12 dialogues at standard (1.0x) and slow learning (0.75x) speeds with `female_v4`, `male_v1`, and `male_zs` in `output_v2/dialogues` and `output_v2_slow/dialogues` with ms JSON timestamps.
- **Dual-Audio Vocabulary (Lessons 7–10):** Extracted 215 clean vocabulary words; synthesized 1.0x and 0.65x dual audio using `female_v4` with both standard and `v3_norm_` / `v3_slow_` naming formats.
- **Verification:** All 62 unit tests passing (100% green). Baseline 345 cards preserved with deterministic GUIDs, 0 collisions across 560 total cards.

---

## 3. Git Release Tags
- `pre-v3` $\rightarrow$ `v3-vad` $\rightarrow$ `v3-female` $\rightarrow$ `v3-male` $\rightarrow$ `v3-dialogues` $\rightarrow$ `v3-examples` $\rightarrow$ `v3-anki-dual` $\rightarrow$ `v3-complete` $\rightarrow$ `pre-v4` $\rightarrow$ `v4-dialogues-text` $\rightarrow$ `v4-male` $\rightarrow$ `v4-male-b` $\rightarrow$ `v4-female` $\rightarrow$ `v4-complete`

---

## 4. Key File Map for Review
| Component | Primary File | Description |
| :--- | :--- | :--- |
| **VAD & Pitch** | `tutor/backend/audio_utils.py` | Speech trimming, F0 contour & YIN pitch |
| **Tone & Sandhi**| `tutor/backend/pronunciation_engine.py`, `sandhi.py` | Syllable alignment, MLP tone scoring |
| **AI Teacher** | `tutor/backend/talk/ws.py`, `teacher_prompt.py` | WebSocket turns, pedagogical prompt |
| **Dialogues** | `scripts/generate_multi_speaker_dialogues.py` | Multi-speaker room-tone stitched audio |
| **Anki Export** | `scripts/export_anki_deck.py` | Dual-audio SRS exporter (`--preserve-model`) |
| **Voice Config**| `configs/voice_profiles.yaml`, `dialogue_speakers.yaml` | Voice mapping (female_v4, female_v3, male_v1, male_zs) |

> **Rollback Instruction (Voice Profile):**
> To revert speaker A to v3, set `A: female_v3` in [`configs/dialogue_speakers.yaml`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/configs/dialogue_speakers.yaml).

---

## 5. Verification Commands
- **Run Tests:** `D:\miniconda3\envs\GPTSoVits\python.exe -m pytest tests/ -v`
- **Verify Git Status:** `git status` and `git tag -l`
