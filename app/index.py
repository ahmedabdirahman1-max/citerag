"""A small, dependency-free TF-IDF vector index over the document chunks.

This is the "vector store + embeddings" of the demo. It's intentionally simple
and transparent: TF-IDF + cosine similarity, pure Python. Swapping in real
embeddings (OpenAI, sentence-transformers) means replacing `_vectorize` and the
similarity call — the retrieval/guardrail/citation logic above it is unchanged.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple

from .corpus import Chunk
from .text import tokenize

Vector = Dict[str, float]


def _l2_normalize(vec: Vector) -> Vector:
    norm = math.sqrt(sum(w * w for w in vec.values()))
    if norm == 0:
        return vec
    return {t: w / norm for t, w in vec.items()}


class TfidfIndex:
    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.idf: Vector = {}
        self._vectors: List[Vector] = []

    def fit(self, chunks: List[Chunk]) -> "TfidfIndex":
        self.chunks = chunks
        n = len(chunks)
        tokenized = [tokenize(c.text) for c in chunks]

        df: Dict[str, int] = {}
        for toks in tokenized:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        # Smoothed idf so a term in every chunk still has non-zero weight.
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}

        self._vectors = [self._vectorize(toks) for toks in tokenized]
        return self

    def _vectorize(self, tokens: List[str]) -> Vector:
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}
        return _l2_normalize(vec)

    def query(self, text: str, k: int) -> List[Tuple[Chunk, float]]:
        q = self._vectorize(tokenize(text))
        scored: List[Tuple[Chunk, float]] = []
        for chunk, vec in zip(self.chunks, self._vectors):
            # cosine = dot product (both vectors are L2-normalized)
            score = sum(w * vec.get(t, 0.0) for t, w in q.items())
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
