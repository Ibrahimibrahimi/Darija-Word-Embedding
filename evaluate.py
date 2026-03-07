"""
evaluate.py
───────────
Query and evaluate a trained Arabic Word2Vec model.

Usage:
    python evaluate.py --vectors models/arabic_vectors.kv
    python evaluate.py --vectors models/arabic_vectors.kv --word كتاب
"""

import argparse
from gensim.models import KeyedVectors


def load_vectors(vectors_path: str) -> KeyedVectors:
    """Load saved word vectors."""
    wv = KeyedVectors.load(vectors_path)
    print(f"[evaluate] Loaded vectors: {len(wv):,} words | dim={wv.vector_size}")
    return wv


def most_similar(wv: KeyedVectors, word: str, topn: int = 10) -> None:
    """Print the most similar words to a given Arabic word."""
    if word not in wv:
        print(f"  [!] Word '{word}' not in vocabulary.")
        return

    print(f"\n أقرب {topn} كلمات لـ '{word}':")
    print("  " + "─" * 35)
    for w, score in wv.most_similar(word, topn=topn):
        bar = "█" * int(score * 20)
        print(f"  {w:20s}  {score:.4f}  {bar}")


def word_analogy(wv: KeyedVectors, pos1: str, pos2: str, neg: str, topn: int = 5) -> None:
    """
    Solve analogy: pos1 - neg + pos2 = ?
    Example: ملك - رجل + امراه = ؟  (king - man + woman = ?)
    """
    words = [pos1, pos2, neg]
    missing = [w for w in words if w not in wv]
    if missing:
        print(f"  [!] Words not in vocabulary: {missing}")
        return

    print(f"\n تشابه المتجهات: {pos1} - {neg} + {pos2} = ؟")
    print("  " + "─" * 35)
    results = wv.most_similar(positive=[pos1, pos2], negative=[neg], topn=topn)
    for w, score in results:
        print(f"  {w:20s}  {score:.4f}")


def word_similarity(wv: KeyedVectors, word1: str, word2: str) -> None:
    """Print cosine similarity between two words."""
    if word1 not in wv or word2 not in wv:
        print(f"  [!] One or both words not in vocabulary.")
        return
    score = wv.similarity(word1, word2)
    print(f"\n التشابه بين '{word1}' و '{word2}': {score:.4f}")


def doesnt_match(wv: KeyedVectors, words: list[str]) -> None:
    """Find the odd word out from a list."""
    in_vocab = [w for w in words if w in wv]
    if len(in_vocab) < 2:
        print("  [!] Not enough words in vocabulary.")
        return
    odd = wv.doesnt_match(in_vocab)
    print(f"\n الكلمة الغريبة في {in_vocab}: '{odd}'")


def run_demo(wv: KeyedVectors) -> None:
    """Run a demonstration of all evaluation functions."""
    print("\n" + "═" * 50)
    print("       Arabic Word2Vec — Evaluation Demo")
    print("═" * 50)

    # Pick the 4th most common word as demo (skip very common stopwords at 0-2)
    demo_words = list(wv.index_to_key)
    demo_word  = demo_words[3] if len(demo_words) > 3 else demo_words[0]

    # 1. Most similar
    most_similar(wv, demo_word, topn=8)

    # 2. Vocabulary stats
    print(f"\n إحصائيات المفردات:")
    print(f"  عدد الكلمات  : {len(wv):,}")
    print(f"  حجم المتجه   : {wv.vector_size}")
    print(f"  أكثر 10 كلمات تكراراً:")
    for w in demo_words[:10]:
        print(f"    - {w}")


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Arabic Word2Vec model")
    parser.add_argument("--vectors", type=str, default="models/arabic_vectors.kv",
                        help="Path to saved .kv vectors file")
    parser.add_argument("--word",    type=str, default=None,
                        help="Arabic word to query (optional)")
    parser.add_argument("--topn",    type=int, default=10,
                        help="Number of similar words to return")
    args = parser.parse_args()

    wv = load_vectors(args.vectors)

    if args.word:
        most_similar(wv, args.word, topn=args.topn)
    else:
        run_demo(wv)
