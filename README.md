# Text Classification: BERT + From-Scratch Transformer + Word2Vec

Three-notebook project building up to a 43-class text classifier — starting from Word2Vec implemented from scratch, through a custom Transformer, up to fine-tuned BERT.

---

## Overview

Rather than jumping straight to BERT, the project builds upward: first implementing Word2Vec skip-gram from scratch to understand embeddings, then building a Transformer classifier from scratch, then fine-tuning pretrained BERT as the final step. Each stage is a separate notebook so the contribution of pretraining is directly measurable.

---

## Notebooks & Results

**`Word2Vec_from_Scratch.ipynb`**
Skip-gram model with negative sampling, implemented in PyTorch from scratch. Generates embeddings, includes similarity checks and nearest-word lookups.

- Val accuracy: **77.97%** | Val F1: **77.90%**

**`Transformer_Text_Classifier.ipynb`**
Transformer-style classifier built from scratch using the Word2Vec embeddings. Sinusoidal positional encodings, multi-head self-attention, feed-forward classification head, end-to-end training.

- Val accuracy: **70.13%** | Val F1: **70.38%**

**`BERT_Classifier.ipynb`**
Fine-tunes pretrained BERT-base on the same task with stop-word removal, lemmatization, and text normalization. Hyperparameter tuning over learning rate, batch size, and epochs.

- Val accuracy: **81.63%** | Val F1: **81.65%**

> Note: GitHub renders BERT_Classifier.ipynb as "Invalid Notebook" due to a widget metadata issue — runs fine locally. Clone the repo or download individually.

---

## Summary

| Model | Val Accuracy | Val F1 |
|-------|-------------|--------|
| Word2Vec (scratch) | 77.97% | 77.90% |
| Transformer (scratch) | 70.13% | 70.38% |
| BERT fine-tuned | 81.63% | 81.65% |

The ~12% gap between scratch Transformer and BERT shows what pretraining contributes on a 43-class problem. Interestingly, the Word2Vec embeddings alone (without attention) outperform the full scratch Transformer — likely because the attention mechanism needs more data or epochs to converge properly.

---

## Task

43-class text classification from `train.csv` (columns: `Text`, `Category`).

---

## Stack

- **PyTorch** — Word2Vec, custom Transformer, training loops
- **HuggingFace Transformers** — BERT fine-tuning
- **NLTK** — preprocessing (stopwords, lemmatization)

---

Saket Vispute, IIT Bombay
