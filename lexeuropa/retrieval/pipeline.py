"""Orchestrates the hybrid retrieval pipeline:

    reference resolver ─┐
    BM25 keyword search ├─► reciprocal rank fusion ─► cross-encoder rerank
    dense (bge-m3) search┘

Chunk text carries no citation label ("Article 9(1) GDPR" lives only in
chunk.citation, not chunk.text), so a templated question like "What does
Article 9(1) of the GDPR provide?" gives BM25/dense/the reranker nothing
to match on — only the resolver reads that metadata. Observed live: the
resolver found the exact chunk, but the reranker's cross-encoder scored
it below other GDPR text and dropped it from the final top-k entirely.
Small, precise resolver hits are pinned below to fix this.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from lexeuropa.config import USE_RERANKER
from lexeuropa.retrieval.bm25_index import BM25Index
from lexeuropa.retrieval.dense_index import DenseIndex
from lexeuropa.retrieval.fusion import reciprocal_rank_fusion
from lexeuropa.retrieval.reference_resolver import ReferenceResolver
from lexeuropa.retrieval.reranker import RerankerUnavailable, rerank
from lexeuropa.schema import Chunk

# A resolver hit is pinned into the final result only when it's a small,
# precise set (an exact "Article N(M)" match) — not a bare case-number
# match, which can return 80-100 paragraphs and would overwhelm k_final.
MAX_PINNED_RESOLVER_HITS = 4


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    scores: dict[str, float]
    reranked: bool


def _ensure_pinned(
    chunks: list[Chunk], scores: dict[str, float], pool: list[Chunk], pinned_ids: set[str]
) -> tuple[list[Chunk], dict[str, float]]:
    """Guarantees every pinned id is present, swapping in for the
    lowest-scored non-pinned slot rather than prepending — a pinned hit
    must not always land in the same (e.g. first) position."""
    chunk_by_id = {c.id: c for c in pool}
    present = {c.id for c in chunks}
    missing = [pid for pid in pinned_ids if pid not in present and pid in chunk_by_id]
    if not missing:
        return chunks, scores

    chunks = list(chunks)
    for pid in missing:
        non_pinned_idx = [i for i, c in enumerate(chunks) if c.id not in pinned_ids]
        if not non_pinned_idx:
            break
        worst_idx = min(non_pinned_idx, key=lambda i: scores.get(chunks[i].id, float("-inf")))
        chunks[worst_idx] = chunk_by_id[pid]
        scores[pid] = scores.get(pid, 0.0)
    return chunks, scores


def retrieve(
    query: str,
    *,
    bm25_index: BM25Index,
    dense_index: DenseIndex,
    resolver: ReferenceResolver,
    k_candidates: int = 20,
    k_final: int = 8,
    default_celex: str | None = None,
    use_reranker: bool | None = None,
) -> RetrievalResult:
    if use_reranker is None:
        use_reranker = USE_RERANKER
    resolved = resolver.resolve(query, default_celex=default_celex)
    precise = resolver.resolve_precise(query, default_celex=default_celex)
    bm25_hits = [c for c, _ in bm25_index.search(query, k_candidates)]
    dense_hits = [c for c, _ in dense_index.search(query, k_candidates)]

    fused = reciprocal_rank_fusion([resolved, bm25_hits, dense_hits])
    fused_chunks = [c for c, _ in fused][:k_candidates]
    # Pinning is gated on the *precise* hit count alone, not resolved's
    # total — resolved also carries a bare case-number's full-paragraph
    # dump (can be 80-100 chunks), which must never block pinning for an
    # exact Article-N match found in the same query. Observed live: a
    # question naming both "Article 17 GDPR" and a bare case number had
    # len(resolved) > MAX_PINNED_RESOLVER_HITS from the case dump alone,
    # which under the old resolved-based gate silently dropped Article 17
    # pinning too, and the reranker then dropped it from the result
    # entirely. See reference_resolver.resolve_precise().
    pinned_ids = {c.id for c in precise} if 0 < len(precise) <= MAX_PINNED_RESOLVER_HITS else set()
    # A pinned chunk might have ranked outside k_candidates in the fused
    # list (unlikely, but RRF doesn't guarantee it) — make sure it's at
    # least in the pool that gets passed to the reranker.
    for c in resolved:
        if pinned_ids and c.id not in {fc.id for fc in fused_chunks}:
            fused_chunks.append(c)

    if not fused_chunks:
        return RetrievalResult([], {}, False)

    try:
        if not use_reranker:
            raise RerankerUnavailable("disabled by caller")
        reranked = rerank(query, fused_chunks, k_final)
        chunks = [c for c, _ in reranked]
        scores = {c.id: s for c, s in reranked}
        if pinned_ids:
            chunks, scores = _ensure_pinned(chunks, scores, fused_chunks, pinned_ids)
        return RetrievalResult(chunks=chunks, scores=scores, reranked=True)
    except RerankerUnavailable as e:
        if use_reranker:
            warnings.warn(f"reranker unavailable, using fused ranking instead: {e}")
        top = fused[:k_final]
        chunks = [c for c, _ in top]
        scores = {c.id: s for c, s in top}
        if pinned_ids:
            chunks, scores = _ensure_pinned(chunks, scores, fused_chunks, pinned_ids)
        return RetrievalResult(chunks=chunks, scores=scores, reranked=False)
