"""Multilingual cross-encoder reranking of fused retrieval candidates."""
from __future__ import annotations

from lexeuropa.config import RERANKER_MODEL
from lexeuropa.schema import Chunk

_model = None


class RerankerUnavailable(RuntimeError):
    """Raised when the cross-encoder can't be loaded (not installed, no
    network to fetch weights, etc). Callers should catch this and fall back
    to the pre-rerank (fused) ordering rather than crash the whole pipeline.
    """


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RerankerUnavailable(
                "sentence-transformers is not installed (pip install sentence-transformers)"
            ) from e
        try:
            # device="cpu" deliberately, not auto-detected MPS — see CLAUDE.md.
            _model = CrossEncoder(RERANKER_MODEL, device="cpu")
        except Exception as e:
            raise RerankerUnavailable(
                f"could not load reranker model {RERANKER_MODEL!r}: {e}"
            ) from e
    return _model


def rerank(query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
    """Raises RerankerUnavailable if the model can't be loaded."""
    if not chunks:
        return []
    model = _get_model()
    pairs = [(query, c.index_text) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])
    return [(c, float(s)) for c, s in ranked[:top_k]]
