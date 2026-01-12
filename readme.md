# Text Classification with BERT & From-Scratch Transformer

## 📒 Notebooks Structure  

| Notebook | Description |
|:-------------|:---------------------------------------------------|
| `BERT_Classifier.ipynb` | Fine-tunes pretrained BERT for text classification. (Note: May show invalid in some environments like GitHub preview, but works fine locally — download individually or clone the repo to run.) |
| `Transformer_Text_Classifier.ipynb` | Implements a Transformer-style classifier using pretrained Word2Vec and attention mechanisms from scratch. |
| `Word2Vec_from_Scratch.ipynb` | Implements Word2Vec skip-gram model with negative sampling from scratch in PyTorch. |

---

## Important Note:

If you encounter any issues previewing the `BERT_Classifier.ipynb` notebook on GitHub o, download the file individually or clone the repo — it runs perfectly in local environments.

## Task  
Build a state-of-the-art text classifier to assign one of **43 labels** from text data.

---

## Implementation Details  

### Fine-tune Pretrained BERT  
- **Dataset:** `train.csv` (with `Category` and `Text` columns)
- **Preprocessing:**
  - Stop-word removal
  - Lemmatization
  - Text normalization
  - Text augmentation (optional)
- **Metrics reported:**
  - Training & validation loss curves
  - Accuracy, precision, recall, and F1 scores
- **Hyperparameter Tuning:**
  - Learning rate
  - Batch size
  - Number of epochs  
  Documented and analyzed their effect on model performance.

---

### Transformer-Style Classifier from Scratch  
- **Pretrained Word2Vec embeddings**
- **Sinusoidal positional encodings**
- **Multi-head self-attention module**
- **Feed-forward classification head**
- End-to-end training and evaluation.

---

### Word2Vec Implementation from Scratch  
- Built **Word2Vec Skip-gram with Negative Sampling** from scratch using PyTorch.
- Generated embeddings.
- Performed similarity checks and nearest word lookups.
---

