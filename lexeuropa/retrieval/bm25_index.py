"""Keyword retrieval.

Tokenization is lowercase Unicode word/number splitting with no stemming
(e.g. no Porter stemmer) — the corpus is English-only for now, but this
keeps the index language-neutral rather than baking in an English-only
assumption that a future non-English corpus would silently degrade under.
"""
from __future__ import annotations

import pickle
import re

from rank_bm25 import BM25Okapi

from lexeuropa.schema import Chunk

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._tokenized = [tokenize(c.index_text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized) if chunks else None

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: -x[1])
        return [(c, float(s)) for c, s in ranked[:k] if s > 0]

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump([c.to_dict() for c in self.chunks], f)

    @classmethod
    def load(cls, path) -> "BM25Index":
        with open(path, "rb") as f:
            chunk_dicts = pickle.load(f)
        return cls([Chunk.from_dict(d) for d in chunk_dicts])
