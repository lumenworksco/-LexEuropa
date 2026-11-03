"""Reciprocal rank fusion for combining BM25 and dense retrieval rankings."""
from __future__ import annotations

from lexeuropa.schema import Chunk


def reciprocal_rank_fusion(
    rankings: list[list[Chunk]], k: int = 60
) -> list[tuple[Chunk, float]]:
    """Combine several ranked chunk lists into one, via RRF.

    score(chunk) = sum over rankings containing chunk of 1 / (k + rank)
    """
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            chunk_by_id[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(scores, key=lambda cid: -scores[cid])
    return [(chunk_by_id[cid], scores[cid]) for cid in ranked_ids]
