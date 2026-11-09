"""Reciprocal rank fusion for combining BM25 and dense retrieval rankings."""
from __future__ import annotations

from lexeuropa.schema import Chunk


def reciprocal_rank_fusion(
    rankings: list[list[Chunk]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[Chunk, float]]:
    """Combine several ranked chunk lists into one, via RRF.

    score(chunk) = sum over rankings containing chunk of weight / (k + rank)

    `weights` (default all 1.0) lets a hint ranking, such as the legal-vocabulary rewrite of
    the question, count for less than the user's own words.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk in enumerate(ranking, start=1):
            chunk_by_id[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (k + rank)

    ranked_ids = sorted(scores, key=lambda cid: -scores[cid])
    return [(chunk_by_id[cid], scores[cid]) for cid in ranked_ids]
