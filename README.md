# Arabic Word2Vec Project

Train a Word2Vec model on Arabic text with full preprocessing.

## Project Structure

```
arabic_word2vec/
│
├── data/
│   └── corpus.txt          ← PUT YOUR ARABIC CORPUS HERE (one sentence per line)
│
├── models/                 ← Auto-created after training
│   ├── arabic_word2vec.model
│   └── arabic_vectors.kv
│
├── outputs/                ← Auto-created after visualization
│   └── tsne_arabic.png
│
├── preprocess.py           ← Arabic normalization & tokenization
├── train.py                ← Word2Vec training
├── evaluate.py             ← Query & evaluate the model
├── visualize.py            ← t-SNE plot
├── run_pipeline.py         ← Run everything at once
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Option A — Run full pipeline
```bash
# train model from corpus to word embeddings
python run_pipeline.py --corpus data/corpus.txt
```

### Option B — Step by step
```bash
# 1. Train
python train.py --corpus data/corpus.txt --sg 1 --epochs 20

# 2. Evaluate
python evaluate.py --vectors models/arabic_vectors.kv --word كتاب

# 3. Visualize
python visualize.py --vectors models/arabic_vectors.kv --topn 150
```

## Corpus Format

Plain `.txt` file, UTF-8 encoded, **one Arabic sentence per line**:


## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `vector_size` | 150 | Increase to 200-300 for large corpus |
| `window` | 5 | Context window around each word |
| `min_count` | 3 | Ignore rare words |
| `sg` | 1 | 1=Skip-gram (recommended for Arabic) |
| `epochs` | 20 | More epochs = better quality |
 بالساهل