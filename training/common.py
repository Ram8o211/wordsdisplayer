"""Shared definitions between training (Python) and inference (TypeScript).

Everything in this file MUST stay bit-exact with `src/nn/text.ts` and
`src/nn/concepts.ts`: tokenisation, FNV-1a hashing, bucket counts and the
concept list. `export.py` writes these constants into the model manifest and
the app refuses to load a model whose constants do not match.
"""
import re
import unicodedata

CONCEPTS = [
    "ocean", "beach", "forest", "mountain", "desert", "city", "space", "night",
    "rain", "snow", "fire", "sunset", "river", "flowers", "storm", "underwater",
    "technology", "music", "love", "clouds", "birds", "fields",
]
NUM_CONCEPTS = len(CONCEPTS)
# Extra regression outputs after the concept logits: valence, arousal in [-1, 1]
NUM_AFFECT = 2

# ---- Traditional network (ANN) constants
ANN_BUCKETS = 8192
ANN_MAX_LEN = 48  # longest context window (ultra tier)

# ---- Neuromorphic network (SNN) constants
SNN_AFFERENTS = 4096
SNN_WORD_SPIKES = 10   # afferents activated by a word
SNN_BIGRAM_SPIKES = 2  # afferents activated by (previous word, word)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def tokenize(text: str):
    return _TOKEN_RE.findall(normalize(text))


def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def ann_ids(tokens):
    """Per position: (unigram bucket, bigram bucket). Bigram of the first
    position is hashed against a beginning-of-stream marker."""
    uni, bi = [], []
    prev = "<s>"
    for t in tokens:
        uni.append(fnv1a("w:" + t) % ANN_BUCKETS)
        bi.append(fnv1a("b:" + prev + "_" + t) % ANN_BUCKETS)
        prev = t
    return uni, bi


def snn_afferents(prev: str, tok: str):
    """Sparse binary code of a word: the afferent indices that spike."""
    idx = set()
    for i in range(SNN_WORD_SPIKES):
        idx.add(fnv1a("s:" + tok + ":" + str(i)) % SNN_AFFERENTS)
    for i in range(SNN_BIGRAM_SPIKES):
        idx.add(fnv1a("p:" + prev + "_" + tok + ":" + str(i)) % SNN_AFFERENTS)
    return sorted(idx)
