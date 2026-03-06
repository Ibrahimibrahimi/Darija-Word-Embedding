"""
run_pipeline.py
───────────────
End-to-end pipeline: preprocess → train → evaluate → visualize.

Usage:
    python run_pipeline.py
    python run_pipeline.py --corpus data/corpus.txt --skip_viz
"""

import argparse
import os

from preprocess import load_corpus
from train import train
from evaluate import load_vectors, most_similar, word_similarity, doesnt_match
from visualize import plot_tsne


def main(corpus_path: str, skip_viz: bool = False):

    print("\n" + "═" * 55)
    print("    Arabic Word2Vec Pipeline — Starting")
    print("═" * 55 + "\n")

    # ── Step 1: Train ────────────────────────────────────────────
    model = train(
        corpus_path=corpus_path,
        model_dir="models",
        vector_size=300,
        window=10,
        min_count=5,
        sg=1,       # Skip-gram
        negative=10,
        epochs=30,
    )

    wv = model.wv

    # ── Step 2: Evaluate ─────────────────────────────────────────
    print("\n" + "─" * 40)
    print(" Step 2: Evaluation")
    print("─" * 40)

    # Use most common word as demo
    demo_words = list(wv.index_to_key)
    if demo_words:
        demo = demo_words[min(3, len(demo_words) - 1)]
        most_similar(wv, demo, topn=8)

    print("\n" + "═" * 55)
    print("    Pipeline complete!")
    print(f"    Model   → models/arabic_word2vec.model")
    print(f"    Vectors → models/arabic_vectors.kv")
    if not skip_viz:
        print(f"    Plot    → outputs/tsne_arabic.png")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",    type=str, default="data/corpus.txt")
    parser.add_argument("--skip_viz",  action="store_true",
                        help="Skip t-SNE visualization")
    args = parser.parse_args()
    main(corpus_path=args.corpus, skip_viz=args.skip_viz)
