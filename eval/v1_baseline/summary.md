# TTS Evaluation Report — v1_baseline

- **Date:** 2026-10-07 18:51:16
- **Total Test Items:** 80 (Evaluated: 80)
- **Overall Mean CER:** **16.88%**
- **Overall Tone Match Rate:** **80.91%**
- **Exact Accuracy (CER = 0):** **57.50%**

---

### Category Performance Breakdown

| Category | Items | Mean CER | Tone Match | Exact Match Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Complex** | 20 | 8.89% | 79.40% | 20.00% |
| **Dialogue** | 30 | 4.09% | 91.17% | 80.00% |
| **Vocab** | 30 | 35.00% | 71.67% | 60.00% |

---

### Top 10 Problematic / Challenging Items

| ID | Category | Target Text | ASR Transcription | CER | Tone Match | Flag |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| V05 | vocab | 一天 | 明镜及点点栏目 | 100.0% | 0.0% | `HIGH_CER` |
| V11 | vocab | 妈 | 哇 | 100.0% | 0.0% | `HIGH_CER` |
| V12 | vocab | 麻 | 拜拜 | 100.0% | 0.0% | `HIGH_CER` |
| V13 | vocab | 马 |  | 100.0% | 0.0% | `HIGH_CER` |
| V14 | vocab | 骂 |  | 100.0% | 0.0% | `HIGH_CER` |
| V18 | vocab | 得去 | 多谢 | 100.0% | 0.0% | `HIGH_CER` |
| V06 | vocab | 不是 | 胡适 | 100.0% | 50.0% | `HIGH_CER` |
| V17 | vocab | 觉得 | 绝大 | 100.0% | 50.0% | `HIGH_CER` |
| V08 | vocab | 妈妈 | 媽媽 | 100.0% | 100.0% | `HIGH_CER` |
| V07 | vocab | 不好 | 多好 | 50.0% | 50.0% | `HIGH_CER` |

---

### Duration & Pronunciation Notes
- Vocab items with 1-2 characters frequently need punctuation padding (e.g. `。`) or best-of-N selection.
- Polyphonic characters and sandhi shifts require accurate G2P normalization.
