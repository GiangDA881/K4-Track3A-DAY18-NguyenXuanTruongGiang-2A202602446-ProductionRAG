from __future__ import annotations

"""Offline fallback encoder — dùng khi không tải được model từ HuggingFace.

Hashing vectorizer trên word unigram + bigram (đã lowercase), L2-normalized.
Chất lượng thấp hơn nhiều so với bge-m3 / MiniLM, chỉ để pipeline vẫn chạy
được trong môi trường không có mạng (CI, sandbox). Luôn in cảnh báo khi dùng.
"""

import hashlib
import re
from itertools import pairwise

import numpy as np

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    words = _TOKEN_RE.findall(text.lower())
    return words + [f"{a}_{b}" for a, b in pairwise(words)]


class HashingEncoder:
    """API tương thích tối thiểu với SentenceTransformer.encode()."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, token: str) -> int:
        return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "little") % self.dim

    def encode(self, texts, show_progress_bar: bool = False, **_kwargs):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        out = np.zeros((len(batch), self.dim), dtype=np.float32)
        for row, text in enumerate(batch):
            for tok in _tokens(text):
                out[row, self._bucket(tok)] += 1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out[0] if single else out


def load_sentence_encoder(model_name: str, dim: int):
    """Load SentenceTransformer; fallback sang HashingEncoder nếu không tải được model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as e:
        print(f"  ⚠️  Không load được '{model_name}' ({type(e).__name__}) — dùng HashingEncoder offline.")
        return HashingEncoder(dim=dim)
