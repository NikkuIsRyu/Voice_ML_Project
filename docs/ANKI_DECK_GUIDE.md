# Anki Vokabel-Decks v3 (Dual-Audio & Kontextuelle Beispielsätze)

Dieses Verzeichnis enthält die fertigen Anki-Decks (`.apkg`) für den Chinesisch-Sprachkurs, vollständig aktualisiert auf **Voice-ML v3**:
- **Dual-Audio auf jeder Karte:** Normales Sprechtempo (`🐇 1.0x`) und didaktisch verlangsamtes, natives Sprechtempo (`🐢 0.65x`), beide eingesprochen von der verfeinerten Studio-Stimme `female_v3` (CSMSC 2.5 Stunden Fine-Tuning).
- **Kontextuelle Beispielsätze (Verbesserung 2.4):** Jede der 345 Vokabeln besitzt einen authentischen Beispielsatz auf HSK1/HSK2-Niveau mit fett hervorgehobenem Zielwort (`<b>...</b>`), ton-farbigem Pinyin, deutscher Übersetzung und abspielbarem Satz-Audio (`💬 Satz`).
- **Zero-Loss In-Place Update (Verbesserung 1.1):** Das Notizmodell und die deterministischen Karten-GUIDs (`genanki.guid_for(lesson_code, str(line_idx), hanzi)`) sind zu **100 % identisch** mit deinen bisherigen Karten. Beim Import in Anki werden alle 345 Notizen **in-place aktualisiert** – **alle deine Lernintervalle, Fälligkeiten und Wiederholungsstatistiken bleiben zu 100 % erhalten!**

---

## 🛡️ Schritt-für-Schritt Backup-Plan vor dem Import

Bevor du das aktualisierte Deck in dein Anki importierst, führe bitte folgende Schritte aus:

### 1. Vollständiges Anki-Sammlungs-Backup (`.colpkg`)
1. Öffne **Anki Desktop**.
2. Klicke im Menü auf **`Datei` ➔ `Exportieren...`**.
3. Wähle als Exportformat: **`Anki-Sammlungspaket (*.colpkg)`**.
4. Stelle sicher, dass die Haken bei **„Zeitplanung einschließen“** und **„Medien einschließen“** gesetzt sind.
5. Klicke auf **`Exportieren...`** und speichere die Datei z. B. unter:
   `backups/Anki_Sammlung_Backup_vor_v3.colpkg`
*(Falls jemals etwas unerwartet aussieht, stellt ein Doppelklick auf diese `.colpkg`-Datei den exakten Stand auf die Sekunde genau wieder her).*

### 2. Automatisches APKG-Backup durch das Skript
Das Skript `scripts/export_anki_deck.py` sichert vor jedem Export automatisch alle bestehenden `.apkg`-Dateien aus `output_v2/anki/` mit Zeitstempel in den Ordner `backups/anki_apkg_backup_<timestamp>/`.

---

## 📥 In-Place Update in Anki Desktop durchführen

1. Öffne **Anki Desktop**.
2. Wähle **`Datei` ➔ `Importieren...`** und wähle die Datei:
   [`output_v2/anki/Chinesisch_Vokabeln_Alle_Lektionen.apkg`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/output_v2/anki/Chinesisch_Vokabeln_Alle_Lektionen.apkg)
3. **Wichtig für Anki 23.10+:** Achte im Import-Dialog auf folgende Einstellung:
   - **Bestehende Notizen aktualisieren:** `Immer` (oder `Wenn neuer`)
4. Klicke auf **`Importieren`**.
5. Anki meldet: *„345 Notizen aktualisiert, 0 Notizen hinzugefügt“*.
   - **Ergebnis:** Deine Karten haben sofort die neue `female_v3`-Stimme, Dual-Audio (`1.0x` + `0.65x`) und Beispielsätze – **bei unverändertem Lernfortschritt!**

---

## 📦 Verfügbare Decks

| Datei | Beschreibung | Vokabeln | Notizen / Karten | Audio-Dateien |
|---|---|---:|---:|---:|
| **`Chinesisch_Vokabeln_Alle_Lektionen.apkg`** | **Master-Deck mit allen Lektionen als Unterdecks** | **345** | **345 Notizen / 690 Karten** | **1.035 Audio-Clips** (345× 1.0x + 345× 0.65x + 345× Beispielsatz) |
| `Lektion_01_Begruessung_und_Kennenlernen.apkg` | Lektion 01: Begrüßung und Kennenlernen | 37 | 37 / 74 | 111 Audio-Clips |
| `Lektion_03_Einkaufen_Kontakte_Orte_und_Bildung.apkg` | Lektion 03: Einkaufen, Kontakte, Orte & Bildung | 86 | 86 / 172 | 258 Audio-Clips |
| `Lektion_04_Uhrzeit_und_Datum.apkg` | Lektion 04: Uhrzeit und Datum | 67 | 67 / 134 | 201 Audio-Clips |
| `Lektion_05_Orte_und_Entfernungen_Buero_und_Richtungen.apkg` | Lektion 05: Orte, Entfernungen, Büro & Richtungen | 66 | 66 / 132 | 198 Audio-Clips |
| `Lektion_06_Kommen_und_Gehen_Reisen_und_Wirtschaft.apkg` | Lektion 06: Kommen/Gehen, Reisen & Wirtschaft | 79 | 79 / 158 | 237 Audio-Clips |
| `Beispiel-Lektion_Grundwortschatz.apkg` | Beispiel-Lektion: Grundwortschatz | 10 | 10 / 20 | 30 Audio-Clips |

---

## 🎴 Karten-Design & Didaktischer Aufbau

Jede Vokabel enthält 2 Kartentypen:

### 1. Karte 1: Lesen & Erkennen (Hanzi ➔ Pinyin, Audio, Bedeutung, Beispielsatz)
- **Vorderseite:** Großes Hanzi + Lektions-Badge.
- **Rückseite:**
  - Ruby-Annotation (Pinyin exakt über den Schriftzeichen)
  - Farbcodiertes Pinyin (Pleco-Farbstandard: 1. Ton Rot, 2. Ton Grün, 3. Ton Blau, 4. Ton Lila, Neutral Grau)
  - Deutsche Übersetzung
  - **Dual-Audio Badges:**
    - `🐇 1.0x` (Standard-Sprechtempo)
    - `🐢 0.65x` (didaktisch verlangsamt, deutliche Artikulation)
  - **Beispielsatz-Box:**
    - Authentischer chinesischer Satz mit fettem Zielwort
    - Pinyin-Umschrift
    - Deutsche Übersetzung
    - `💬 Satz` Audio-Abspielbutton

### 2. Karte 2: Hören & Verstehen (Audio ➔ Hanzi, Pinyin, Bedeutung, Beispielsatz)
- **Vorderseite:** Spielt Audio ab, Kopfhörer-Symbol 🎧, Hinweis-Button für deutsche Bedeutung.
- **Rückseite:** Vollständige Auflösung mit Dual-Audio, Schriftzeichen, farbigem Pinyin und Beispielsatz.

---

## 🛠️ Befehl zum erneuten Exportieren

```powershell
D:\miniconda3\envs\GPTSoVits\python.exe scripts/export_anki_deck.py --cards-mode both
```
