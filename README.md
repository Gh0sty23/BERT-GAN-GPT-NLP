# Comparative NLP: BERT, GPT-2, and a Custom GAN for Tweet Sentiment & Generation

A hands-on comparison of three major deep learning paradigms — transformer-based classification (BERT), autoregressive generation (GPT-2), and adversarial generation (a custom LSTM-based GAN) — applied to a tweet sentiment dataset.

---

## Overview

Understanding *why* different NLP architectures behave differently isn't just about reading their papers — it's about putting them side by side on the same data and watching where each one succeeds or breaks down. This project fine-tunes **BERT** for sentiment classification, fine-tunes **GPT-2** for text generation, and trains a **GAN from scratch** for text generation, all on the same tweet corpus, then compares them on tokenization strategy, training behavior, and evaluation metrics.

The goal isn't to crown a "best" model — it's to surface the practical tradeoffs (tokenization scheme, differentiability, evaluation metric fit) that make each architecture suited to a different kind of task.

---

## Approach

### Data
- **Source**: Tweet sentiment classification dataset (`sahideseker/tweet-sentiment-classification-dataset`, via Kaggle)
- **Labels**: 3-class sentiment — `neutral`, `negative`, `positive`
- Tweets are also written out to a flat `.txt` corpus for GPT-2 fine-tuning and GAN training

### 1. BERT — Sentiment Classification
- `bert-base-uncased` fine-tuned via `BertForSequenceClassification` (Hugging Face)
- WordPiece tokenization (`BertTokenizer`, `max_length=64`, `[CLS]`/`[SEP]` special tokens)
- 90/10 train/validation split, batch size 32
- Optimizer: `AdamW` (lr `2e-5`, eps `1e-8`) with a linear warmup/decay schedule
- 4 training epochs, evaluated each epoch

### 2. GPT-2 — Text Generation
- `gpt-2-simple` (124M parameter GPT-2) fine-tuned on the raw tweet corpus for 500 steps
- Byte Pair Encoding (BPE) tokenization
- Generation sampled at `temperature=0.7`, evaluated with corpus BLEU (BLEU-1 through BLEU-4)

### 3. Custom GAN — Text Generation
- **Generator**: LSTM → linear projection over the BERT vocabulary, sampled with `argmax`
- **Discriminator**: LSTM → linear → sigmoid (real vs. fake classifier)
- Trained adversarially with `BCELoss` and Adam (lr `0.0002`) for 10 epochs
- Shares the BERT `BertTokenizer` vocabulary for encoding real data and decoding generated output
- Evaluated with the same corpus BLEU metrics as GPT-2

---

## Tech Stack

| Category | Library |
|---|---|
| Language | Python 3 |
| Environment | Google Colab (GPU runtime), Jupyter Notebook |
| Transformers | `transformers` (`BertTokenizer`, `BertForSequenceClassification`) |
| Text generation | `gpt-2-simple`, `tensorflow` |
| Deep learning | `torch` (LSTM Generator/Discriminator, custom training loop) |
| Data handling | `pandas`, `numpy`, `opendatasets`, `kagglehub` |
| Evaluation | `scikit-learn` (F1, precision, recall), `nltk` (BLEU / `corpus_bleu`) |
| Visualization | `matplotlib`, `seaborn` |

---

## Results

| Model | Task | Key Metric(s) | Result |
|---|---|---|---|
| **BERT** | Sentiment classification | Validation Accuracy / F1 / Precision / Recall | **1.00** across all four (see note below) |
| **GPT-2** | Text generation | Qualitative fluency | Coherent, grammatical, tweet-style sentences |
| **Custom GAN** | Text generation | BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4 | 0.0027 / 0.0002 / 0.0001 / 0.0001 |

> **Note on BERT's perfect score:** A validation accuracy/F1/precision/recall of 1.00 is a red flag, not a triumph. It most likely reflects a small, low-diversity dataset with limited class overlap rather than genuine generalization — in a production setting this would warrant closer inspection for data leakage or an unrepresentative validation split.

> **Note on the GAN's low BLEU scores:** The near-zero BLEU scores are expected, not a bug. Text is discrete, which makes gradients from the discriminator hard to propagate to the generator — this implementation uses non-differentiable `argmax` sampling rather than a workaround like Gumbel-Softmax or REINFORCE, so the generator never learns to produce coherent sequences.

---

## Key Takeaways

- **Tokenization is not interchangeable across architectures.** BERT's WordPiece tokenizer is built for robust subword representation (good for classification); GPT-2's BPE is built for fluent open-ended generation; reusing BERT's tokenizer inside the GAN was a convenience choice, not a natural fit for generation.
- **Autoregressive models (GPT-2) are structurally better suited to open-ended text generation than adversarial models (GAN) in a discrete token space**, because GPT-2 directly models next-token likelihood while the GAN's generator has no clean gradient signal from the discriminator.
- **A metric is only as good as its fit to the task.** BLEU rewards n-gram overlap with references, which flatters likelihood-based generators (GPT-2) and unfairly punishes GANs that might generate valid but non-matching text — though in this case, the GAN's output was genuinely incoherent, so the low score is deserved.
- **A perfect classification score should trigger scrutiny, not celebration.** It's a signal to check for data leakage or an overly easy validation split before trusting the model.

---
