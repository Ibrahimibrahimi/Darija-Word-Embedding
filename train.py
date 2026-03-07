"""
train.py
────────
Train a Word2Vec model on an Arabic text corpus.

Usage:
    python train.py --corpus data/corpus.txt
    python train.py --corpus data/corpus.txt --sg 1 --epochs 20 --size 150
"""

import argparse
import multiprocessing
import os
from gensim.models import Word2Vec
from preprocess import load_corpus


def train(
    corpus_path: str,
    model_dir: str   = "models",
    vector_size: int = 150,
    window: int      = 5,
    min_count: int   = 3,
    sg: int          = 1,
    negative: int    = 10,
    epochs: int      = 20,
    seed: int        = 42,
) -> Word2Vec:
    """
    Train and save a Word2Vec model.

    Args:
        corpus_path : Path to Arabic .txt corpus (one sentence per line)
        model_dir   : Directory to save model files
        vector_size : Dimensionality of word vectors (100–300)
        window      : Context window size
        min_count   : Ignore words with freq < min_count
        sg          : 0 = CBOW  |  1 = Skip-gram (recommended for Arabic)
        negative    : Number of negative samples
        epochs      : Training iterations
        seed        : Random seed for reproducibility

    Returns:
        Trained gensim Word2Vec model
    """
    # ── 1. Load & tokenize corpus ────────────────────────────────
    sentences = load_corpus(corpus_path)

    if len(sentences) == 0:
        raise ValueError("Corpus is empty after preprocessing. Check your file.")

    # ── 2. Train ─────────────────────────────────────────────────
    print(f"\n[train] Starting Word2Vec training...")
    print(f"        Algorithm   : {'Skip-gram' if sg == 1 else 'CBOW'}")
    print(f"        Vector size : {vector_size}")
    print(f"        Window      : {window}")
    print(f"        Min count   : {min_count}")
    print(f"        Epochs      : {epochs}")
    print(f"        Workers     : {multiprocessing.cpu_count()}")

    model = Word2Vec(
        sentences   = sentences,
        vector_size = vector_size,
        window      = window,
        min_count   = min_count,
        workers     = multiprocessing.cpu_count(),
        sg          = sg,
        hs          = 0,             # use negative sampling
        negative    = negative,
        epochs      = epochs,
        seed        = seed,
    )

    # ── 3. Save ──────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)

    model_path   = os.path.join(model_dir, "arabic_word2vec.model")
    vectors_path = os.path.join(model_dir, "arabic_vectors.kv")

    model.save(model_path)
    model.wv.save(vectors_path)

    print(f"\n[train] ✓ Training complete!")
    print(f"        Vocabulary size : {len(model.wv):,} words")
    print(f"        Model saved     : {model_path}")
    print(f"        Vectors saved   : {vectors_path}")

    return model


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Arabic Word2Vec model")
    parser.add_argument("--corpus",   type=str, default="data/corpus.txt", help="Path to corpus file")
    parser.add_argument("--model_dir",type=str, default="models",          help="Directory to save model")
    parser.add_argument("--size",     type=int, default=150,               help="Vector dimensionality")
    parser.add_argument("--window",   type=int, default=5,                 help="Context window size")
    parser.add_argument("--min_count",type=int, default=3,                 help="Minimum word frequency")
    parser.add_argument("--sg",       type=int, default=1,                 help="1=Skip-gram, 0=CBOW")
    parser.add_argument("--epochs",   type=int, default=20,                help="Training epochs")
    args = parser.parse_args()

    train(
        corpus_path = args.corpus,
        model_dir   = args.model_dir,
        vector_size = args.size,
        window      = args.window,
        min_count   = args.min_count,
        sg          = args.sg,
        epochs      = args.epochs,
    )
