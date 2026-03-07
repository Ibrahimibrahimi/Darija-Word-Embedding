"""
preprocess.py
─────────────
Arabic text normalization and tokenization utilities.
Used by train.py and evaluate.py
"""

import re

try:
    import pyarabic.araby as araby
    USE_PYARABIC = True
except ImportError:
    USE_PYARABIC = False
    print("[WARNING] pyarabic not installed. Falling back to regex-only normalization.")


# ── Arabic Unicode ranges & patterns ────────────────────────────

DIACRITICS_PATTERN = re.compile(
    r"[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)
TATWEEL_PATTERN = re.compile(r"\u0640")          # ـ
NON_ARABIC_PATTERN = re.compile(r"[^\u0600-\u06FF\s]")
MULTI_SPACE = re.compile(r"\s+")


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics (Tashkeel)."""
    if USE_PYARABIC:
        return araby.strip_tashkeel(text)
    return DIACRITICS_PATTERN.sub("", text)


def normalize_arabic(text: str) -> str:
    """
    Full normalization pipeline:
      1. Strip diacritics
      2. Strip Tatweel (elongation)
      3. Normalize Alef variants  → ا
      4. Normalize Teh Marbuta    → ه
      5. Normalize Yeh variants   → ي
      6. Remove non-Arabic chars
      7. Collapse whitespace
    """
    # 1. Diacritics
    if USE_PYARABIC:
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
    else:
        text = DIACRITICS_PATTERN.sub("", text)
        text = TATWEEL_PATTERN.sub("", text)

    # 2. Alef normalization (أ إ آ → ا)
    text = re.sub(r"[إأآ]", "ا", text)

    # 3. Teh Marbuta (ة → ه)
    text = re.sub(r"ة", "ه", text)

    # 4. Yeh normalization (ى → ي)
    text = re.sub(r"ى", "ي", text)

    # 5. Remove non-Arabic
    text = NON_ARABIC_PATTERN.sub(" ", text)

    # 6. Collapse spaces
    text = MULTI_SPACE.sub(" ", text).strip()

    return text


def tokenize(text: str, min_token_len: int = 2) -> list[str]:
    """
    Normalize and split a sentence into Arabic word tokens.
    Filters tokens shorter than min_token_len.
    """
    normalized = normalize_arabic(text)
    tokens = normalized.split()
    return [t for t in tokens if len(t) >= min_token_len]


def load_corpus(filepath: str, min_sentence_len: int = 3) -> list[list[str]]:
    """
    Load a plain-text Arabic corpus (one sentence per line).
    Returns a list of tokenized sentences.

    Args:
        filepath        : Path to .txt corpus file (UTF-8)
        min_sentence_len: Skip sentences with fewer tokens than this

    Returns:
        List of token lists, e.g. [["الكتاب", "جميل"], ...]
    """
    sentences = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            tokens = tokenize(line)

            if len(tokens) < min_sentence_len:
                continue

            sentences.append(tokens)

    print(
        f"[preprocess] Loaded {len(sentences):,} sentences from '{filepath}'")
    print(f"[preprocess] Total tokens: {sum(len(s) for s in sentences):,}")
    return sentences


# ── Quick test ───────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "وَالْكِتَابُ الجَمِيلُ مَوجُودٌ عَلى الطَّاوِلَةِ.",
        "الأطفال يلعبون في الحديقة الجميلة",
        "مرحباً! كيف حالك؟ أتمنى أن تكون بخير.",
    ]

    print("=== Normalization Test ===")
    for s in samples:
        print(f"  Input : {s}")
        print(f"  Output: {normalize_arabic(s)}")
        print(f"  Tokens: {tokenize(s)}")
        print()
