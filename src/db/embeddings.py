"""Embedding engine for The Associate.

Adapted from HMLR's EmbeddingManager pattern. Wraps sentence-transformers
for encode/cosine_similarity and provides pgvector-compatible storage.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.core.config import EmbeddingsConfig

logger = logging.getLogger("associate.embeddings")

# Lazy import — sentence-transformers is heavy
_SentenceTransformer = None


def _get_sentence_transformer():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


class EmbeddingEngine:
    """Encodes text to vectors and provides similarity search utilities.

    Uses all-MiniLM-L6-v2 (384 dimensions) by default.
    Model is loaded lazily on first encode() call.
    """

    def __init__(self, config: Optional[EmbeddingsConfig] = None):
        self.config = config or EmbeddingsConfig()
        self.model_name = self.config.model
        self.dimension = self.config.dimension
        self._model = None

    @property
    def model(self):
        if self._model is None:
            SentenceTransformer = _get_sentence_transformer()
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded (%dD)", self.dimension)
        return self._model

    def encode(self, text: str) -> list[float]:
        """Encode a single text string to a vector.

        Returns a plain Python list of floats (compatible with pgvector).
        """
        vec = self.model.encode(text, show_progress_bar=False)
        return vec.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts to vectors."""
        vecs = self.model.encode(texts, show_progress_bar=False, batch_size=32)
        return [v.tolist() for v in vecs]

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns a value between -1 and 1 (1 = identical).
        """
        a = np.array(vec1)
        b = np.array(vec2)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    @staticmethod
    def to_pgvector_str(vec: list[float]) -> str:
        """Convert a vector to pgvector's text format: '[0.1,0.2,...]'"""
        return "[" + ",".join(str(v) for v in vec) + "]"
