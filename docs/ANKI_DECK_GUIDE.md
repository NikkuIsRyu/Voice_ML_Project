# Anki Vokabel-Decks mit Refined Audio (CSMSC v2ProPlus)

Dieses Verzeichnis enthält die fertigen Anki-Decks (`.apkg`) für den Chinesisch-Sprachkurs, vollständig vertont mit der aktualisierten, verfeinerten Stimme (**Biaobei / CSMSC v2ProPlus Modell**).

---

## 📦 Verfügbare Decks

### 1. 🐢 Verlangsamte Version (Natives Neural-TTS Tempo – Natürlich & Deutlich)
Synthetisiert direkt durch das **GPT-SoVITS Modell mit `speed_factor=0.65`** (kein unnatürliches Audio-Stretching, sondern echte verlangsamte menschliche Artikulation).
Gespeichert in [`output_v2_slow/anki/`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/output_v2_slow/anki/):

| Datei | Lektion | Vokabeln | Karten | Audio-Dauer | Größe |
|---|---|---:|---:|---:|---:|
| **`Chinesisch_Vokabeln_Alle_Lektionen_Langsam_Neural.apkg`** | **Alle Lektionen (Master-Deck mit Unterdecks)** | **345** | **690** | **9.6 Min** | **~35.4 MB** |
| `Lektion_01_Begruessung_und_Kennenlernen.apkg` | Lektion 01: Begrüßung und Kennenlernen | 37 | 74 | 1.0 Min | ~3.4 MB |
| `Lektion_03_Einkaufen_Kontakte_Orte_und_Bildung.apkg` | Lektion 03: Einkaufen, Kontakte, Orte & Bildung | 86 | 172 | 2.5 Min | ~9.2 MB |
| `Lektion_04_Uhrzeit_und_Datum.apkg` | Lektion 04: Uhrzeit und Datum | 67 | 134 | 1.9 Min | ~7.3 MB |
| `Lektion_05_Orte_und_Entfernungen_Buero_und_Richtungen.apkg` | Lektion 05: Orte, Entfernungen, Büro & Richtungen | 66 | 132 | 1.8 Min | ~6.6 MB |
| `Lektion_06_Kommen_und_Gehen_Reisen_und_Wirtschaft.apkg` | Lektion 06: Kommen/Gehen, Reisen & Wirtschaft | 79 | 158 | 2.2 Min | ~8.0 MB |
| `Beispiel-Lektion_Grundwortschatz.apkg` | Beispiel-Lektion: Grundwortschatz | 10 | 20 | 0.3 Min | ~1.2 MB |

### 2. ⚡ Standard-Version (100% Original-Tempo)
Gespeichert in [`output_v2/anki/`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/output_v2/anki/):

| Datei | Lektion | Vokabeln | Karten | Audio-Dauer | Größe |
|---|---|---:|---:|---:|---:|
| **`Chinesisch_Vokabeln_Alle_Lektionen.apkg`** | **Alle Lektionen (Master-Deck mit Unterdecks)** | **345** | **690** | **6.8 Min** | **~25.2 MB** |

---

## 🎴 Deck-Hierarchie & Struktur

Das Master-Deck `Chinesisch_Vokabeln_Alle_Lektionen.apkg` legt in Anki automatisch eine übersichtliche Baumstruktur mit Unterdecks an:

```text
▾ Chinesisch Sprachkurs (Refined Voice)
    ▸ Lektion 01 - Begrüßung und Kennenlernen
    ▸ Lektion 03 - Einkaufen, Kontakte, Orte und Bildung
    ▸ Lektion 04 - Uhrzeit und Datum
    ▸ Lektion 05 - Orte und Entfernungen, Büro und Richtungen
    ▸ Lektion 06 - Kommen und Gehen, Reisen und Wirtschaft
    ▸ Beispiel-Lektion - Grundwortschatz
```

- **Alle Lektionen gemeinsam lernen:** Klicke einfach auf das Hauptdeck `Chinesisch Sprachkurs (Refined Voice)`.
- **Gezielt eine Lektion wiederholen:** Klicke direkt auf das jeweilige Unterdeck (z. B. `Lektion 01`).

---

## 🧠 Zwei Kartentypen pro Vokabel

Jeder Vokabeleintrag erzeugt 2 didaktisch optimierte Kartentypen:

### 1. Karte 1: Lesen & Erkennen (Hanzi ➔ Pinyin, Audio, Bedeutung)
- **Vorderseite:** Großes chinesisches Schriftzeichen (Hanzi) + Lektions-Badge.
- **Rückseite:** 
  - Schriftzeichen mit **Ruby-Annotation** (Pinyin direkt über den Zeichen).
  - Farbcodiertes Pinyin nach Tönen.
  - Deutsche Übersetzung.
  - **Automatische Sprachausgabe** der verfeinerten Stimme (`[sound:...wav]`).
  - Ton-Farblegende zur Orientierung.

### 2. Karte 2: Hören & Verstehen (Audio ➔ Hanzi, Pinyin, Bedeutung)
- **Vorderseite:** 
  - Die Audio-Aussprache spielt sofort ab.
  - Großes Kopfhörer-Symbol 🎧.
  - Frage: *"Welches Wort wird gesprochen?"*
  - Optionaler Hinweis-Button: *"Tipp: Deutsche Bedeutung einblenden"*.
- **Rückseite:** Auflösung mit Schriftzeichen, farbigem Pinyin, deutscher Übersetzung und Audio-Wiederholung.

> **Tipp:** Wenn du nur Lese- oder nur Hörkarten lernen möchtest, kannst du in Anki über die Kartenverwaltung (*Browser*) einfach den entsprechenden Kartentyp mit einem Klick pausieren (*Suspend*).

---

## 🎨 Ton-Farben (Pleco / Lernstandard)

Die Pinyin-Silben sind konsistent nach den standardisierten Mandarin-Tönen eingefärbt:

- <span style="color:#e11d48; font-weight:bold;">● 1. Ton (Hoch, eben):</span> Rot (`#e11d48`)
- <span style="color:#16a34a; font-weight:bold;">● 2. Ton (Steigend):</span> Grün (`#16a34a`)
- <span style="color:#2563eb; font-weight:bold;">● 3. Ton (Fallend-steigend):</span> Blau (`#2563eb`)
- <span style="color:#9333ea; font-weight:bold;">● 4. Ton (Fallend):</span> Violett (`#9333ea`)
- <span style="color:#64748b; font-weight:bold;">● Neutraler Ton:</span> Schiefergrau (`#64748b`)

*Das Design passt sich automatisch an den Anki Dark Mode (Night Mode) an!*

---

## 📥 Import-Anleitung

1. **Anki Desktop öffnen** (oder AnkiMobile / AnkiDroid).
2. Doppelklicke auf [`output_v2/anki/Chinesisch_Vokabeln_Alle_Lektionen.apkg`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/output_v2/anki/Chinesisch_Vokabeln_Alle_Lektionen.apkg) oder wähle in Anki **Datei ➔ Importieren**.
3. Anki importiert alle 345 Audio-Dateien direkt in die Mediendatenbank (`collection.media`). Es wird keine externe Verbindung oder laufender Server benötigt!
4. Fertig! Loslernen mit perfekter nativer Audio-Aussprache.

---

## 🛠️ Erneutes Generieren / Anpassen

Das Skript [`scripts/export_anki_deck.py`](file:///d:/.D%20Dokuments/Uni%20Stuff/Meine_Bibliothek/Code_Projects/Voice_ML/Voice_ML_Project/scripts/export_anki_deck.py) kann jederzeit erneut ausgeführt werden:

```bash
# Beide Kartentypen (Standard)
python scripts/export_anki_deck.py

# Nur Lese- und Erkennungskarten:
python scripts/export_anki_deck.py --cards-mode recognition

# Nur Hörverständniskarten:
python scripts/export_anki_deck.py --cards-mode listening
```
