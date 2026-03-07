"""
visualize.py
────────────
t-SNE 2D visualization of Arabic Word2Vec embeddings.

Usage:
    python visualize.py --vectors models/arabic_vectors.kv
    python visualize.py --vectors models/arabic_vectors.kv --topn 200 --out outputs/tsne.png
"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors

# Use non-interactive backend (safe for servers)
matplotlib.use("Agg")

# ── Font setup (Arabic-compatible) ──────────────────────────────
# Try to use a system font that supports Arabic glyphs.
# Falls back to DejaVu Sans if none found.
ARABIC_FONTS = ["Amiri", "Scheherazade New", "Arial Unicode MS", "DejaVu Sans"]

def set_arabic_font():
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for font in ARABIC_FONTS:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            return font
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans (fallback — Arabic may not render)"


def plot_tsne(
    wv: KeyedVectors,
    topn: int      = 150,
    perplexity: int = 30,
    out_path: str  = "outputs/tsne_arabic.png",
) -> None:
    """
    Run t-SNE on top-N word vectors and save a scatter plot.

    Args:
        wv         : Loaded KeyedVectors
        topn       : Number of most-frequent words to plot
        perplexity : t-SNE perplexity (5–50, lower = tighter clusters)
        out_path   : Where to save the PNG
    """
    font = set_arabic_font()
    print(f"[visualize] Font used: {font}")

    # ── Collect vectors ──────────────────────────────────────────
    topn    = min(topn, len(wv))
    words   = list(wv.index_to_key[:topn])
    vectors = np.array([wv[w] for w in words])

    print(f"[visualize] Running t-SNE on {topn} words (dim={vectors.shape[1]})...")

    # ── t-SNE ────────────────────────────────────────────────────
    tsne = TSNE(
        n_components = 2,
        perplexity   = min(perplexity, topn - 1),
        n_iter       = 1000,
        random_state = 42,
        init         = "pca",
        learning_rate= "auto",
    )
    coords = tsne.fit_transform(vectors)

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 13))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=25, alpha=0.7,
        c=coords[:, 0],        # color by x-position for visual variety
        cmap="plasma",
        zorder=2,
    )

    for i, word in enumerate(words):
        ax.annotate(
            word,
            xy       = (coords[i, 0], coords[i, 1]),
            fontsize = 7,
            ha       = "right",
            color    = "#e0e0e0",
            alpha    = 0.85,
        )

    ax.set_title(
        "تمثيل الكلمات العربية — Word2Vec + t-SNE",
        fontsize = 15,
        color    = "white",
        pad      = 15,
    )
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"[visualize] ✓ Plot saved to '{out_path}'")


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE visualization of Arabic Word2Vec")
    parser.add_argument("--vectors",    type=str, default="models/arabic_vectors.kv")
    parser.add_argument("--topn",       type=int, default=150, help="Number of words to plot")
    parser.add_argument("--perplexity", type=int, default=30,  help="t-SNE perplexity")
    parser.add_argument("--out",        type=str, default="outputs/tsne_arabic.png")
    args = parser.parse_args()

    wv = KeyedVectors.load(args.vectors)
    print(f"[visualize] Loaded {len(wv):,} vectors")

    plot_tsne(wv, topn=args.topn, perplexity=args.perplexity, out_path=args.out)
