"""Dense lexical vectors for candidate retrieval.

name and address are embedded separately: hashed char 3-gram TF-IDF -> TruncatedSVD ->
L2-normalise. The retrieval vector concatenates both halves scaled by sqrt(w) so a dot
product equals ``w_name * cos_name + w_addr * cos_addr``. An empty address yields a zero
half, so such records are retrieved on name alone.

Fitted without labels on a sample of all sources (train + test), so it also sees French
text. No pretrained model or external data is involved.
"""
import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

# Normalised text is ASCII a-z0-9 + space, so there are at most 37**3 ~ 50k char 3-grams.
N_FEATURES = 2 ** 17
DIM = 128
W_NAME, W_ADDR = 0.5, 0.5


def _hasher():
    return HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), n_features=N_FEATURES,
        alternate_sign=False, norm=None, dtype=np.float32,
    )


class FieldEmbedder:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=4, random_state=0)

    def fit(self, texts):
        x = self.tfidf.fit_transform(_hasher().transform(texts))
        self.svd.fit(normalize(x))
        self.svd.components_ = self.svd.components_.astype(np.float32)  # halves worker RAM
        return self

    def sparse(self, texts):
        """L2-normalised TF-IDF rows (used for exact cosine reranking)."""
        return normalize(self.tfidf.transform(_hasher().transform(texts)))

    def dense(self, texts) -> np.ndarray:
        z = self.svd.transform(self.sparse(texts)).astype(np.float32)
        return normalize(z).astype(np.float16)  # all-zero rows stay zero


class RecordEmbedder:
    def __init__(self):
        self.name = FieldEmbedder()
        self.addr = FieldEmbedder()

    def fit(self, names, addrs):
        self.name.fit(names)
        self.addr.fit([a for a in addrs if a])
        return self

    def dense(self, names, addrs) -> np.ndarray:
        n = self.name.dense(names) * np.float16(np.sqrt(W_NAME))
        a = self.addr.dense(addrs) * np.float16(np.sqrt(W_ADDR))
        return np.hstack([n, a])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "RecordEmbedder":
        with open(path, "rb") as f:
            return pickle.load(f)
