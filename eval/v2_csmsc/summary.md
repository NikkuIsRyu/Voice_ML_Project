# TTS Evaluation Report — v2_csmsc

- **Date:** 2026-10-07 22:26:09
- **Total Test Items:** 80 (Evaluated: 80)
- **Overall Mean CER:** **12.85%**
- **Overall Tone Match Rate:** **84.80%**
- **Exact Accuracy (CER = 0):** **68.75%**

---

### Category Performance Breakdown

| Category | Items | Mean CER | Tone Match | Exact Match Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Complex** | 20 | 10.22% | 81.50% | 35.00% |
| **Dialogue** | 30 | 3.02% | 93.47% | 86.67% |
| **Vocab** | 30 | 24.44% | 78.33% | 73.33% |

---

### Top 10 Problematic / Challenging Items

| ID | Category | Target Text | ASR Transcription | CER | Tone Match | Flag |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| V11 | vocab | 妈 |  | 100.0% | 0.0% | `HIGH_CER` |
| V12 | vocab | 麻 | 媽 | 100.0% | 0.0% | `HIGH_CER` |
| V13 | vocab | 马 | 嗎? | 100.0% | 0.0% | `HIGH_CER` |
| V14 | vocab | 骂 | 媽 | 100.0% | 0.0% | `HIGH_CER` |
| V23 | vocab | 首都 | 少多 | 100.0% | 0.0% | `HIGH_CER` |
| V17 | vocab | 觉得 | 绝大 | 100.0% | 50.0% | `HIGH_CER` |
| V27 | vocab | 出租车 | 丑座车 | 66.7% | 33.3% | `HIGH_CER` |
| V25 | vocab | 小城市 | 小同事 | 66.7% | 66.7% | `HIGH_CER` |
| L01 | complex | 他的电话号码是一三八零零一三八零零零，如果联系不上可以发微信。 | 他的电话号码是138000,如果联系不上可以发微信。 | 37.9% | 24.1% | `HIGH_CER` |
| D27 | dialogue | 一百五十块钱，质量非常好。 | 150块钱,质量非常好。 | 36.4% | 0.0% | `HIGH_CER` |

---

### Duration & Pronunciation Notes
- Vocab items with 1-2 characters frequently need punctuation padding (e.g. `。`) or best-of-N selection.
- Polyphonic characters and sandhi shifts require accurate G2P normalization.
