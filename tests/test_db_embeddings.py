"""Tests for src/db/embeddings.py — embedding engine.

Requires sentence-transformers installed. Uses real model loading.
"""

import pytest

from src.core.config import EmbeddingsConfig
from src.db.embeddings import EmbeddingEngine


def _sentence_transformers_available() -> bool:
    try:
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        return False


requires_sentence_transformers = pytest.mark.skipif(
    not _sentence_transformers_available(),
    reason="sentence-transformers not installed",
)


@requires_sentence_transformers
class TestEmbeddingEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        """Shared engine — model loads once per class (it's expensive)."""
        return EmbeddingEngine(EmbeddingsConfig())

    def test_encode_returns_384d_vector(self, engine):
        vec = engine.encode("Hello world")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_encode_different_texts_differ(self, engine):
        vec1 = engine.encode("Python programming language")
        vec2 = engine.encode("Italian pizza recipe")
        assert vec1 != vec2

    def test_encode_batch(self, engine):
        texts = ["Hello", "World", "Test"]
        vecs = engine.encode_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_cosine_similarity_identical(self, engine):
        vec = engine.encode("test text")
        sim = EmbeddingEngine.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.01

    def test_cosine_similarity_related(self, engine):
        vec1 = engine.encode("Python programming")
        vec2 = engine.encode("Python coding language")
        sim = EmbeddingEngine.cosine_similarity(vec1, vec2)
        assert sim > 0.5  # Related concepts should have decent similarity

    def test_cosine_similarity_unrelated(self, engine):
        vec1 = engine.encode("quantum physics research")
        vec2 = engine.encode("chocolate cake recipe")
        sim = EmbeddingEngine.cosine_similarity(vec1, vec2)
        assert sim < 0.5  # Unrelated topics should have low similarity

    def test_cosine_similarity_zero_vector(self):
        vec1 = [0.0] * 384
        vec2 = [1.0] * 384
        sim = EmbeddingEngine.cosine_similarity(vec1, vec2)
        assert sim == 0.0

    def test_to_pgvector_str(self):
        vec = [0.1, 0.2, 0.3]
        result = EmbeddingEngine.to_pgvector_str(vec)
        assert result == "[0.1,0.2,0.3]"

    def test_to_pgvector_str_empty(self):
        assert EmbeddingEngine.to_pgvector_str([]) == "[]"

    def test_config_defaults(self, engine):
        assert engine.model_name == "all-MiniLM-L6-v2"
        assert engine.dimension == 384

    def test_lazy_model_loading(self):
        """Model should not load until first encode() call."""
        engine = EmbeddingEngine(EmbeddingsConfig())
        assert engine._model is None
        engine.encode("trigger load")
        assert engine._model is not None
