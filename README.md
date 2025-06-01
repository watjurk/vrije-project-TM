# NLP Task Comparison

This project compares models across three NLP tasks: **Sentiment Analysis**, **Topic Classification**, and **Named Entity Recognition Classification (NERC)**. We evaluate how training data and model architecture affect performance using public datasets and pre-trained models.

## Tasks & Models

### Sentiment (18 test phrase samples)
- Models: Bertweet, Twitter-RoBERTa, mBERT-1to5, DistilBERT SST-2, DeBERTa-v3 (Zero-Shot), BART-MNLI (ZS)
- Key Result: DeBERTa-v3 and Bertweet outperform others, especially on neutral sentiment.

### Topic (IMDB, Amazon, Sports)
- Models: Naive Bayes, XGBoost, Neural Net, BART
- Key Result: Naive Bayes has best F1 (0.78); all models struggle with Book vs. Movie due to vocabulary overlap.

### NER (15 test sentences)
- Models: CRF (trained), BERT (dslim), spaCy (ZS)
- Key Result: BERT significantly outperforms CRF, especially on complex entity types (e.g., artworks).

## Highlights
- Zero-shot models show strong performance with minimal tuning.
- Domain-specific training data boosts model accuracy.
- Contextual embeddings (BERT) provide large gains in NER.

## Data Sources
- IMDB, Amazon, scikit-learn 20News, Kaggle NER corpus


