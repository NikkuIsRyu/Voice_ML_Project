# Tone Classifier Evaluation Report

- **Model:** Multilayer Perceptron (128, 64, ReLU)
- **Dataset:** CSMSC (BZNSYP) Phone & Tone Interval Labels (1500 utterances)
- **Total Syllables:** 20908
- **Test Set Accuracy:** **83.24%** (Target >= 85%)

## Classification Breakdown

| Tone | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Tone 1 | 87.1% | 86.5% | 86.8% | 875.0 |
| Tone 2 | 86.4% | 89.6% | 88.0% | 1115.0 |
| Tone 3 | 81.3% | 79.1% | 80.2% | 746.0 |
| Tone 4 | 82.7% | 86.4% | 84.5% | 1231.0 |
| Tone 5 | 49.0% | 33.0% | 39.4% | 215.0 |

## Confusion Matrix (Rows: Actual Tone, Columns: Predicted Tone)

```
[[757, 52, 9, 52, 5], [36, 999, 42, 25, 13], [6, 32, 590, 89, 29], [55, 26, 59, 1064, 27], [15, 47, 26, 56, 71]]
```
