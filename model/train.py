import os
import csv
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .config import (
    EMBED_DIM, WINDOW_SIZE, MIN_COUNT, NEG_SAMPLES,
    BATCH_SIZE, EPOCHS, LR,
    MODEL_PATH, VOCAB_PATH, EMBEDDING_PATH, DATA_PATH
)


# ── Model ─────────────────────────────────────────────────────────────────────

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center  = nn.Embedding(vocab_size, embed_dim)
        self.context = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context, negatives):
        vc = self.center(center)
        vp = self.context(context)
        vn = self.context(negatives)

        pos = torch.sum(vc * vp, dim=1)
        neg = torch.bmm(vn, vc.unsqueeze(2)).squeeze(2)

        loss = -torch.log(torch.sigmoid(pos) + 1e-8)
        loss = loss - torch.sum(torch.log(torch.sigmoid(-neg) + 1e-8), dim=1)
        return loss.mean()

    def get_vector(self, word_id):
        # Return the embedding vector for one word.
        with torch.no_grad():
            return self.center(torch.tensor(word_id)).numpy()


# ── Internal helpers ──────────────────────────────────────────────────────────

class _PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        c, ctx = self.pairs[idx]
        return torch.tensor(c), torch.tensor(ctx)


def _build_vocab(sentences):
    counter = Counter(w for sent in sentences for w in sent)
    vocab   = [w for w, c in counter.items() if c >= MIN_COUNT]
    w2i     = {w: i for i, w in enumerate(vocab)}
    i2w     = {i: w for w, i in w2i.items()}
    return w2i, i2w


def _build_pairs(sentences, w2i):
    pairs = []
    for sent in sentences:
        ids = [w2i[w] for w in sent if w in w2i]
        for i, center in enumerate(ids):
            for j in range(max(0, i - WINDOW_SIZE), min(len(ids), i + WINDOW_SIZE + 1)):
                if j != i:
                    pairs.append((center, ids[j]))
    return pairs


def _load_model():
    # Load saved vocab and model weights from disk.
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    w2i, i2w = vocab["w2i"], vocab["i2w"]
    model = SkipGram(len(w2i), EMBED_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model, w2i, i2w


def _sentence_to_vector(sentence, model, w2i):
    # Average the word vectors in a sentence into one vector.
    tokens = sentence.strip().split() if isinstance(sentence, str) else sentence
    vecs   = [model.get_vector(w2i[w]) for w in tokens if w in w2i]
    return np.mean(vecs, axis=0) if vecs else None


def _cosine_sim(a, b):
    return float((a @ b) / ((a @ a) ** 0.5 * (b @ b) ** 0.5 + 1e-8))


# ── Public API ────────────────────────────────────────────────────────────────

def train(data_path=None):
    """
    Train the Skip-gram model on a text file.
    Each line in the file is one sentence.
    Saves model and vocab to paths defined in config.py.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    path = data_path or DATA_PATH
    with open(path, "r", encoding="utf-8") as f:
        sentences = [line.strip().split() for line in f if line.strip()]
    print(f"Sentences: {len(sentences)}")

    w2i, i2w = _build_vocab(sentences)
    pairs     = _build_pairs(sentences, w2i)
    print(f"Vocab: {len(w2i)} | Pairs: {len(pairs)}")

    loader = DataLoader(_PairDataset(pairs), batch_size=BATCH_SIZE, shuffle=True)
    model  = SkipGram(len(w2i), EMBED_DIM).to(device)
    opt    = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        total = 0
        for center, context in loader:
            center    = center.to(device)
            context   = context.to(device)
            negatives = torch.randint(0, len(w2i), (center.size(0), NEG_SAMPLES)).to(device)
            loss = model(center, context, negatives)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch}/{EPOCHS} — Loss: {total / len(loader):.4f}")

    # Save model weights and vocabulary.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump({"w2i": w2i, "i2w": i2w}, f)
    print(f"Saved to {MODEL_PATH} and {VOCAB_PATH}")


def embed_dataset(data_path=None, output_path=None):
    """
    Convert each sentence in the dataset to a vector and save to a CSV file.
    Each row in the CSV is the embedding of one sentence.
    """
    model, w2i, _ = _load_model()

    path = data_path or DATA_PATH
    with open(path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"Embedding {len(sentences)} sentences...")

    out = output_path or EMBEDDING_PATH
    os.makedirs(os.path.dirname(out), exist_ok=True)
    written = 0

    with open(out, "w", newline="", encoding="utf-8") as csvfile:
        writer = None
        for sent in sentences:
            vec = _sentence_to_vector(sent, model, w2i)
            if vec is None:
                continue
            if writer is None:
                writer = csv.writer(csvfile)
                writer.writerow([f"dim_{i}" for i in range(len(vec))])
            writer.writerow([f"{v:.6f}" for v in vec])
            written += 1

    print(f"Saved {written} vectors to {out}")


def predict_next_word(input_sentence, length=5):
    """
    Predict the next words for a given input sentence.
    Returns a list of (word, score) tuples sorted by similarity.
    Change 'length' to control how many predictions are returned.
    """
    model, w2i, i2w = _load_model()

    query_vec = _sentence_to_vector(input_sentence, model, w2i)
    if query_vec is None:
        print("No known words found in the input.")
        return []

    # Find the most similar words not already in the input.
    input_words = set(input_sentence.strip().split())
    scores = [
        (i2w[idx], _cosine_sim(query_vec, model.get_vector(idx)))
        for idx in i2w if i2w[idx] not in input_words
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:length]
