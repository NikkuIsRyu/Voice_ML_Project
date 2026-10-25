# Phase 0 – Daten-Inventur & Preflight Report

**Datum & Zeit:** 2026-10-25 10:57 CET  
**Projekt:** Voice_ML_Project (v4 Voice-Upgrade)  
**Hardware & System:** Windows 11, Intel CPU, NVIDIA RTX 4060 Ti (8 GB VRAM), D:\ Freier Speicher: 35.18 GB.

---

## 1. Preflight Checks
- **Git Status:** Clean (`main` branch, synchron mit origin). Tag `pre-v4` erfolgreich gesetzt.
- **Aktive Server (Ports 8000, 9880):** Keine aktiven Prozesse blockieren die Ports.
- **VRAM & Speicher:** GPU frei, DataLoader `num_workers=0` vorbereitet.

---

## 2. Datenbestand CSMSC (Biaobei BZNSYP - Female Studio Voice)
- **Gesamtumfang BZNSYP:** 10.000 WAV-Dateien (Studioaufnahme 48 kHz, saubere Prosodie-Annotationen).
- **Bereits in v3 genutzt (`data/train_v3/csmsc`):** 2.468 Clips = genau 2,50 Stunden (9.002,5 s).
- **Verbleibend & ungenutzt in BZNSYP:** 7.532 Clips (~5,4 bis 7,6 Stunden reine Sprachzeit).
- **Verfügbarkeit für Phase 5 (`female_v4`):**
  - **Ziel:** 7,50 Stunden (2,50 h bisher + 5,00 h neu).
  - **Status:** **VOLL VERFÜGBAR**. Die 5,00 h zusätzliche Daten (ca. 4.900 neue Clips) können vollständig aus den ungenutzten BZNSYP-Aufnahmen extrahiert werden.

---

## 3. Datenbestand AISHELL-3 & THCHS30 (Male Stimmen)

### A. Hauptstimme `male_v1` (SSB0817 + multi-speaker male subset)
- **Status in v3:** Trainiert auf 1.478 Clips (1,41 h) aus 6 männlichen AISHELL-3-Sprechern (SSB0817, SSB0863, SSB1187, SSB1239, SSB1328, SSB1365).
- **Referenzstimme:** SSB0817 (`male_v1_ref.wav`, "希望有更多优秀队员上来充实中国游泳的实力。").
- **Verbleibende ungenutzte Daten:**
  - SSB0817 besitzt im gesamten AISHELL-3-Korpus nur 269 Äußerungen (~14 Minuten gesamt). 228 davon wurden in v3 bereits extrahiert (13,2 Minuten). Es verbleiben lediglich 41 Äußerungen (~2,4 Minuten).
  - Auch über alle 6 bisherigen Sprecher summiert verbleiben weniger als 15 Minuten ungenutzte Aufnahmen.
- **Handoff-Regel:**
  > *„Sind < 2 h übrig → alles Verfügbare nehmen, wenn es ≥ 20 min sind. Sind es < 20 min → Phase 3 überspringen, male_v1 bleibt aktiv.“*
- **Entscheidung gemäß Handoff-Gate:**
  - **PHASE 3 WIRD ÜBERSPRUNGEN.**
  - `male_v1` bleibt zu 100% erhalten und aktiv. Dadurch wird die hervorragende, vom Nutzer gelobte Stimmqualität ("sehr gut und sehr echt") garantiert geschützt und vor Überanpassung/Degradierung bewahrt.

### B. Zweitstimme `male_zs` / `male_b`
- **Aktueller Stand `male_zs`:** Verwendet SSB1630 als Zero-Shot-Referenz (`data/refs/male_zs/male_zs_ref.wav`).
- **Verfügbare Daten SSB1630:** Im gesamten AISHELL-3 existieren nur 179 Äußerungen (~10,4 Minuten).
- **Korpus-Scan aller männlichen Sprecher:**
  - In AISHELL-3 existieren 42 männliche Sprecher. Der sprecherreichste männliche Sprecher (SSB0736) hat 489 Äußerungen (~22,2 Minuten). Kein einziger männlicher Sprecher erreicht 30 Minuten.
  - In THCHS30 existieren nur 2 männliche Sprecher mit 2,4 min bzw. 1,0 min.
- **Handoff-Regel:**
  > *„Gibt es keinen mit ≥ 1 h → den mit der meisten Dauer nehmen (mindestens 30 min). Unter 30 min → Phase 4 überspringen, male_zs bleibt.“*
- **Entscheidung gemäß Handoff-Gate:**
  - **PHASE 4 WIRD ÜBERSPRUNGEN.**
  - `male_zs` bleibt als Zero-Shot-Profil erhalten.

---

## 4. Zusammenfassung des Ausführungsplans für v4

1. **Phase 1 (Backups):** Vollständige Sicherung aller bestehenden Modelle (`male_v1`, `female_v3`, `male_zs`) mit SHA256-Prüfsummen.
2. **Phase 2 (Dialoge):**
   - 2a: Bestehende Dialoge (Lektionen 2, 3, 4) inhaltlich komplett umformulieren (neue Szenarien, andere Namen/Orte, Lehrbuch-Originalität eliminieren).
   - 2b: Urheberrechts-/Originalitäts-Check (max. 60% N-Gramm-Überlappung).
   - 2c: Neue Dialoge für die nächsten 3 Lektionen (Lektionen 5, 6, 7) mit jeweils 8–14 Turns auf 3 Sprecher erstellen.
   - Tag: `v4-dialogues-text`.
3. **Phasen 3 & 4:** Übersprungen gemäß den Sicherheitsregeln der Daten-Inventur (`male_v1` und `male_zs` bleiben unverändert erhalten).
4. **Phase 5 (`female_v4`):**
   - Extraktion von 5,00 Stunden neuen BZNSYP-Daten (Gesamt: 7,50 Stunden).
   - Feature-Extraktion (RoBERTa BERT, HuBERT 32k, Semantic Tokens).
   - Fine-Tuning SoVITS (10 Epochen) + GPT (12 Epochen) auf v2ProPlus.
   - Tag: `v4-female`.
5. **Phase 6 (Evaluation & Aktivierungs-Gate):**
   - Synthese des 12-Satz-Testsatzes (`female_v3` vs. `female_v4`).
   - Whisper large-v3 CER-Messung und Outlier-Prüfung.
   - A/B-Samples nach `output/v4_ab_samples/`.
   - Konfigurationsupdate bei bestandenem Gate.
6. **Phase 7 (Dialog-Audio neu erzeugen):**
   - Synthese aller Dialoge (umformulierte L2–L4 sowie neue L5–L7) mit Millisekunden-Timestamps via `scripts/generate_multi_speaker_dialogues.py`.
7. **Phase 8 (Tests, Doku, Git, Bericht):**
   - Pytest Suite (alle Tests grün).
   - Update `CLAUDE.md` (Milestone D).
   - Tag `v4-complete` und Push.
8. **Phase 9 (Herunterfahren):**
   - `shutdown /s /t 120`.
