# 🧠 Multilingual Sarcasm Detection using Transformer Models

Fine-tuning of XLM-RoBERTa, mBERT and DistilBERT for multilingual text classification (Arabic, English, French)

---

## 📌 Project Overview

This project builds a multilingual sarcasm detection system by fine-tuning three Transformer models on a dataset of 60,000+ examples across Arabic, English and French languages.

The best result was achieved using an **Ensemble Learning (Soft Voting)** approach combining all three models, reaching **89% accuracy** and **+12 pts F1-score** over classical baselines.

---

## 🏆 Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression (baseline) | 71% | 0.69 |
| Linear SVM (baseline) | 74% | 0.70 |
| DistilBERT | 80% | 0.78 |
| mBERT | 83% | 0.80 |
| XLM-RoBERTa | 85% | 0.83 |
| **Ensemble (Soft Voting)** | **87%** | **0.85** |

---

## 📂 Dataset

| Source | Language | Size |
|--------|----------|------|
| ar_sarcasm (HuggingFace) | Arabic | tweets |
| sarcasm_headlines_multilingual (HuggingFace) | EN, FR, DE | news headlines |

- **Total after merge & deduplication :** 60,000+ examples
- **Split :** 80% train / 20% test
- **Classes :** 2 (sarcastic / not sarcastic)
- **Max tokens :** 128

---

## ⚙️ Methodology

### Preprocessing
- Arabic : URL/mention/hashtag removal, emoji removal, tashkeel removal, normalization
- EN/FR : URL/mention/hashtag removal, emoji removal, whitespace normalization
- Common : empty texts and duplicates removed

### Models
- `xlm-roberta-base` — 125M parameters, 100 languages
- `bert-base-multilingual-cased` — 110M parameters, 104 languages
- `distilbert-multilingual` — 66M parameters, 104 languages

### Training
- WeightedTrainer with CrossEntropyLoss (class imbalance handling)
- Best checkpoint saved on F1-score

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 2 |
| Weight decay | 0.01 |
| Optimizer | AdamW |

### Ensemble Learning
Soft Voting across 3 models :
