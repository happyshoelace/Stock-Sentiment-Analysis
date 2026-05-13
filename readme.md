# StockSense — Stock Sentiment Analysis Using Social Media

A machine learning system that analyses financial sentiment from social media posts (Twitter/Reddit) using a custom self-training NLP pipeline. No manually labelled data required.

Built by Aashna Gaikwad, Radhika Verma, Bianca Boldeman, and Aryaman Sharma.

---

## Overview

StockSense applies NLP and machine learning to social media discussions to predict whether public sentiment around a given stock is **positive**, **neutral**, or **negative**. Rather than relying on expensive hand-labelled datasets, the system uses a **financial lexicon + iterative self-training** pipeline to bootstrap its own labels from raw, unlabelled tweets.

The dataset covers **25 stock tickers** including TSLA, AMZN and APPL, drawn from over **80,000 real tweets**.

---

## How It Works

The pipeline runs in three phases:

**Phase 1 — Warmup Training**
Tweets are scored using a domain-specific financial lexicon (bullish/bearish word lists, negation handling, cashtag normalisation). High-confidence tweets (≥80% dominant polarity) are used to train the model from scratch.

**Phase 2 — Self-Training (up to 10 rounds)**
The partially trained model runs inference on the unlabelled pool. Any tweet predicted with ≥80% softmax confidence is pseudo-labelled and added to the training set. The model retrains for 3 epochs per round. Used tweets are removed from the pool.

**Phase 3 — Final Fine-Tuning**
The model pseudo-labels the full dataset at a relaxed threshold (≥60%) and fine-tunes for 8 more epochs on this expanded set.

### Model Architecture: `SentimentEnsemble`

A fully custom architecture trained from scratch (no pre-trained weights):

- 3-layer Bidirectional LSTM (hidden size 384)
- 2-layer Bidirectional GRU (hidden size 256)
- 4-head multi-head attention pooling
- Shared embedding layer (dim 256)
- Classification head with GELU activations, dropout (0.40), and layer norm

Training uses AdamW with weight decay, OneCycleLR scheduling, label smoothing (0.10), gradient clipping, and early stopping (patience = 4 epochs).

---

## Repository Structure

```
├── Twitter Dataset/              # Raw tweet data
├── analysis_results/             # Per-ticker sentiment output
├── frontend/                     # Web UI for exploring results
├── stockMentionsTWITTER.json     # Main dataset: ticker → list of tweets
├── stockMentions.json            # Reddit post cache
├── postCache.JSON                # Collected post cache
│
├── create_dataset.py             # Data preparation and formatting
├── twitterDatasetParser.py       # Parses the Twitter JSON into flat dataframes
├── postCollection.py             # Reddit/Twitter post collection
├── documentCollation.py          # Document aggregation utilities
│
├── train_model.py                # Main training script (3-phase pipeline)
├── predict_sentiment.py          # Single-text inference utility
├── batch_predict_json.py         # Batch inference over the full dataset
├── analyze_stocks.py             # Per-ticker sentiment aggregation & stats
├── evaluate_accuracy.py          # Evaluation on held-out pseudo-labelled split
│
├── create_visualizations.py      # Confusion matrix, accuracy curves, charts
├── visualize_twitter_validation.py
├── test_teacher.py               # Prototype: FinBERT → DistilBERT approach
│
├── train_dataset.csv             # Training split
├── val_dataset.csv               # Validation split
├── stockabbreviations.csv        # Ticker symbol reference
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (CUDA recommended for training)

### Installation

```bash
git clone https://github.com/happyshoelace/Stock-Sentiment-Analysis.git
cd Stock-Sentiment-Analysis
pip install -r requirements.txt
```

**Dependencies:** `torch`, `transformers`, `pandas`, `scikit-learn`, `tqdm`, `datasets`

### Training

```bash
python train_model.py
```

This runs the full 3-phase self-training pipeline on `stockMentionsTWITTER.json`.

### Inference

Predict sentiment for a single piece of text:

```bash
python predict_sentiment.py
```

Run batch inference over the full dataset:

```bash
python batch_predict_json.py
```

### Analysis & Visualisation

```bash
python analyze_stocks.py           # Aggregate sentiment stats per ticker
python create_visualizations.py    # Generate confusion matrix and accuracy plots
```

---

## Results

Evaluation is performed on a stratified 15% held-out split of the high-confidence lexicon-labelled pool (not used in any training phase). Note: test labels are lexicon-derived, not human-annotated — metrics reflect how well the model generalises the lexicon signal.

The production model shows a more balanced three-class distribution and less "fence-sitting" compared to the earlier FinBERT → DistilBERT prototype. Qualitative inference tests on clearly bullish, bearish, and ambiguous posts were correct in all evaluated examples, with confidence scores appropriate to the strength of sentiment expressed.

---

## Methodology Notes

Two approaches were explored:

**Approach 1 (prototype):** Use FinBERT to generate pseudo-labels, then fine-tune DistilBERT on those labels. Produced reasonable results but risked inheriting FinBERT's biases and tended toward neutral predictions.

**Approach 2 (production):** Financial lexicon scoring + iterative self-training with a custom architecture trained from scratch. Avoids dependence on any external model and is better suited to the informal vocabulary of social media.

A planned third approach would combine the self-training pipeline with a domain-adapted base model like FinBERT or BERTweet, merging the strengths of both.

---

## Known Limitations

- Lexicon scoring cannot detect sarcasm or irony, which propagates through self-training
- Reddit/Twitter discussions may include misinformation, coordinated pump-and-dump activity, or bot-generated content
- Test set labels are pseudo-labels, not ground truth — reported metrics are a proxy for generalisation, not absolute accuracy
