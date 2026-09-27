"""Learned retrieval encoder: hashed char-3-gram / word / number bags -> two-tower MLP.

Trained from scratch on the provided ground-truth pairs only (no pretrained weights,
no external data). Name and address are pooled separately so an empty address simply
contributes a zero vector.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import HashingVectorizer

N_BUCKETS = 2 ** 20


# --------------------------------------------------------------------------- features
def _word_grams(prefix_w: str, prefix_c: str, text: str) -> list[str]:
    feats = []
    for w in text.split():
        feats.append(prefix_w + w)
        s = f" {w} "
        feats.extend(prefix_c + s[i:i + 3] for i in range(len(s) - 2))
    return feats


def name_features(rec: tuple) -> list[str]:
    name_core, = rec
    return _word_grams("w|", "c|", name_core)


def addr_features(rec: tuple) -> list[str]:
    addr_norm, numbers, states = rec
    words = " ".join(w for w in addr_norm.split() if not w.isdigit())
    return (_word_grams("aw|", "ac|", words)
            + ["n|" + n for n in numbers.split()]
            + ["s|" + s for s in states.split()])


def _hasher(analyzer):
    return HashingVectorizer(analyzer=analyzer, n_features=N_BUCKETS, alternate_sign=False,
                             norm=None, binary=True, dtype=np.float32)


def featurize(name_core, addr_norm, numbers, states):
    """Return (name_csr, addr_csr) with hashed binary features."""
    n = _hasher(name_features).transform(zip(name_core))
    a = _hasher(addr_features).transform(zip(addr_norm, numbers, states))
    return n, a


class CSRBags:
    """Row-gatherable CSR index lists (indptr int64, indices int32) for one field."""

    def __init__(self, indptr: np.ndarray, indices: np.ndarray):
        self.indptr, self.indices = indptr, indices

    def gather(self, rows: np.ndarray):
        starts, ends = self.indptr[rows], self.indptr[rows + 1]
        lens = ends - starts
        offsets = np.zeros(len(rows), dtype=np.int64)
        np.cumsum(lens[:-1], out=offsets[1:])
        total = int(lens.sum())
        # vectorised concatenation of the row slices
        pos = np.repeat(starts - offsets, lens) + np.arange(total)
        return self.indices[pos], offsets, lens


# --------------------------------------------------------------------------- model
class RecordEncoder(nn.Module):
    def __init__(self, n_buckets: int = N_BUCKETS, dim: int = 128, out: int = 128):
        super().__init__()
        self.emb = nn.EmbeddingBag(n_buckets, dim, mode="sum", sparse=True)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 512), nn.GELU(), nn.Linear(512, out))
        self.skip = nn.Linear(2 * dim, out, bias=False)
        self.log_temp = nn.Parameter(torch.tensor(np.log(0.05), dtype=torch.float32))

    def _bag(self, idx, off, lens):
        v = self.emb(idx, off)
        return v / lens.clamp(min=1).float().sqrt().unsqueeze(1)

    def forward(self, name, addr):
        x = torch.cat([self._bag(*name), self._bag(*addr)], 1)
        return F.normalize(self.mlp(x) + self.skip(x), dim=1)


def to_device(bag, device):
    idx, off, lens = bag
    return (torch.from_numpy(idx.astype(np.int64)).to(device, non_blocking=True),
            torch.from_numpy(off).to(device, non_blocking=True),
            torch.from_numpy(lens).to(device, non_blocking=True))
