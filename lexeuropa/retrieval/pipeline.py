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

import os
import warnings
from dataclasses import dataclass

from lexeuropa.config import EXPAND_ARTICLES, EXPAND_RULINGS, EXPAND_SAME_INSTRUMENT, MAX_RECITALS_IN_RESULT, MAX_SOURCES, USE_RERANKER
from lexeuropa.corpus_status import INSTITUTION_QUESTION_RE, is_institution_only, status_of
from lexeuropa.retrieval.bm25_index import BM25Index
from lexeuropa.retrieval.dense_index import DenseIndex
from lexeuropa.retrieval.fusion import reciprocal_rank_fusion
from lexeuropa.retrieval.reference_resolver import ReferenceResolver
from lexeuropa.retrieval.reranker import RerankerUnavailable, rerank
from lexeuropa.schema import para_sort_key, Chunk

# A resolver hit is pinned into the final result only when it's a small,
# precise set (an exact "Article N(M)" match) — not a bare case-number
# match, which can return 80-100 paragraphs and would overwhelm k_final.
MAX_PINNED_RESOLVER_HITS = 4
# When a question is restricted to one statute, at most this many case-law paragraphs may accompany it.
MAX_CASE_LAW_WHEN_RESTRICTED = 2
# Extra slots (beyond MAX_SOURCES) for the rulings of the top judgments.
MAX_RULING_CHUNKS = 4
# Pick the articles to expand by number of ranked hits instead of first appearance.
EXPAND_BY_HITS = os.environ.get("LEXEUROPA_EXPAND_BY_HITS", "1") == "1"
# Soft boost for the statute that leads the top results when no law is named (0 = off).
ROUTE_BOOST = float(os.environ.get("LEXEUROPA_ROUTE_BOOST", "0"))
# Weight of the query-rewrite rankings in fusion, relative to the original question's (1.0).
REWRITE_WEIGHT = float(os.environ.get("LEXEUROPA_REWRITE_WEIGHT", "0.5"))


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


def _cap_recitals(items: list, k: int, max_recitals: int, chunk_of=lambda x: x) -> list:
    """First `k` items in order, but at most `max_recitals` of them recitals.

    Recitals are prose about *why* a rule exists and match natural-language
    questions better than the operative articles do, so left alone they crowd
    the articles out of a small result list (observed: the top 8 for "What
    risk-management measures must entities take under NIS 2?" were all
    recitals, with Article 21 ranked 9th). If there are too few non-recitals
    to fill `k`, the skipped recitals backfill at the end, in order.

    Tried and removed: also capping paragraphs per judgment (max 3 or 2). It
    changed neither hit@8 nor MRR on either question set."""
    out: list = []
    skipped: list = []
    recitals = 0
    for item in items:
        if chunk_of(item).unit_type == "recital":
            if recitals >= max_recitals:
                skipped.append(item)
                continue
            recitals += 1
        out.append(item)
        if len(out) >= k:
            return out
    return (out + skipped)[:k]


def expand_articles(
    chunks: list[Chunk], all_chunks: list[Chunk], n_articles: int, max_sources: int, same_instrument: bool = False,
    by_hits: bool = False,
) -> list[Chunk]:
    """Append the other paragraphs of the top articles after the ranked results.

    An answer to "what are the conditions / which items ..." needs the whole article,
    not whichever of its paragraphs happened to rank. The first `n_articles` distinct
    articles in the ranked results get their remaining paragraphs (statutory order)
    appended, up to `max_sources` in all. The ranked results themselves are never
    reordered or displaced: an earlier version that put the expanded articles first
    pushed the right article out of the list for 3 of 25 questions and produced a
    confidently wrong penalty (the neighbouring article's 3% / EUR 15 million instead
    of the prohibited-practices 7% / EUR 35 million), and made the small model copy
    only the first paragraph it saw. Measured with the grader as the reader, the best
    achievable completeness from the retrieved text rises from 0.72 to 0.83."""
    if n_articles <= 0:
        return chunks
    by_article: dict[tuple[str, str], list[Chunk]] = {}
    for c in all_chunks:
        if c.unit_type == "article_paragraph" and c.article is not None:
            by_article.setdefault((c.celex, c.article), []).append(c)
    top: list[tuple[str, str]] = []
    dominant = next((c.celex for c in chunks if c.unit_type == "article_paragraph"), None)
    for c in chunks:
        key = (c.celex, c.article)
        if c.unit_type == "article_paragraph" and key not in top:
            if same_instrument and c.celex != dominant:
                continue   # a look-alike provision of another regime: do not expand it
            top.append(key)
    if by_hits:
        # An article with several ranked paragraphs (e.g. DMA Art. 3(1) and 3(8)) is more likely the
        # one the question is about than an article with a single hit that merely ranked first.
        hits = {k: sum(1 for c in chunks if (c.celex, c.article) == k and c.unit_type == "article_paragraph") for k in top}
        top = sorted(top, key=lambda k: (-hits[k], top.index(k)))
    out = list(chunks)
    seen = {c.id for c in out}
    for key in top[:n_articles]:
        siblings = sorted(by_article.get(key, []), key=lambda c: para_sort_key(c.para_index))
        for c in siblings[:10]:
            if c.id not in seen and len(out) < max_sources:
                out.append(c)
                seen.add(c.id)
    return out


def expand_rulings(chunks: list[Chunk], all_chunks: list[Chunk], n_cases: int, max_sources: int) -> list[Chunk]:
    """Append the ruling (operative part) of the first `n_cases` judgments in the results.

    "What did the Court decide in Schrems II?" is answered by the operative part ("the Privacy
    Shield Decision is invalid"), which a paragraph-level search over 200+ body paragraphs rarely
    surfaces. The ranked results are never reordered or displaced; rulings are appended."""
    if n_cases <= 0:
        return chunks
    cases: list[str] = []
    for c in chunks:
        if c.unit_type in ("judgment_paragraph", "judgment_operative") and c.celex not in cases:
            cases.append(c.celex)
    out = list(chunks)
    seen = {c.id for c in out}
    for celex in cases[:n_cases]:
        ops = sorted(
            (c for c in all_chunks if c.celex == celex and c.unit_type == "judgment_operative"),
            key=lambda c: int(c.para_index[2:]) if c.para_index[2:].isdigit() else 0,
        )
        for c in ops:
            if c.id not in seen and len(out) < max_sources:
                out.append(c)
                seen.add(c.id)
    return out


def route_by_instrument(
    fused: list[tuple[Chunk, float]], boost: float, top_m: int = 40, min_share: float = 0.25
) -> list[tuple[Chunk, float]]:
    """Soft routing for questions that name no law. With 80+ instruments, look-alike provisions
    ("access to data", "requests for information") scatter the top results over many acts. The
    statute that holds the largest share of the top-`top_m` score mass is probably the one the
    question is about; its chunks get their score multiplied by (1 + boost). Nothing is dropped
    or restricted, and nothing happens unless one statute clearly leads (>= `min_share` of the
    mass), so questions that span several laws are left alone. Judgments are not boosted."""
    if boost <= 0 or not fused:
        return fused
    mass: dict[str, float] = {}
    for c, sc in fused[:top_m]:
        if c.unit_type != "judgment_paragraph" and c.unit_type != "judgment_operative":
            mass[c.celex] = mass.get(c.celex, 0.0) + sc
    total = sum(mass.values())
    if not mass or total <= 0:
        return fused
    leader, lead_mass = max(mass.items(), key=lambda kv: kv[1])
    if lead_mass / total < min_share:
        return fused
    boosted = [(c, sc * (1 + boost) if c.celex == leader else sc) for c, sc in fused]
    boosted.sort(key=lambda x: -x[1])
    return boosted


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
    expand: int | None = None,
    restrict_celex: str | None = None,
    extra_queries: tuple[str, ...] = (),
    route_boost: float | None = None,
) -> RetrievalResult:
    if use_reranker is None:
        use_reranker = USE_RERANKER
    n_expand = EXPAND_ARTICLES if expand is None else expand
    resolved = resolver.resolve(query, default_celex=default_celex)
    precise = resolver.resolve_precise(query, default_celex=default_celex)
    # Search deeper than the final pool so operative articles ranked below a
    # screenful of recitals can still be selected (see _cap_recitals).
    depth = max(k_candidates * 2, 40)
    if restrict_celex:
        depth = 400   # most of a deep search is other instruments; look far enough to find this one's
    bm25_hits = [c for c, _ in bm25_index.search(query, depth)]
    dense_hits = [c for c, _ in dense_index.search(query, depth)]

    rankings = [resolved, bm25_hits, dense_hits]
    weights = [1.0, 1.0, 1.0]
    for extra in extra_queries:   # e.g. the question rewritten in the legal texts' own vocabulary
        # A hint, not a replacement: the rewrite can name the wrong law (it turned a Data Act
        # question about "connected products" into a GDPR "data subject" query and displaced every
        # Data Act result), so its rankings count for less than the user's own words.
        rankings.append([c for c, _ in bm25_index.search(extra, depth)])
        rankings.append([c for c, _ in dense_index.search(extra, depth)])
        weights += [REWRITE_WEIGHT, REWRITE_WEIGHT]
    fused = reciprocal_rank_fusion(rankings, weights=weights)
    if not restrict_celex:
        fused = route_by_instrument(fused, ROUTE_BOOST if route_boost is None else route_boost)
    # Out-of-date and institution-only provisions are skipped unless the question asks for them:
    # repealed law (Directive 95/46, e-Commerce Arts 12-15) is only wanted when named, and
    # provisions for EU institutions (e.g. the AI Act's fines on Union bodies) only when asked.
    named_here = resolver.instruments_named(query)
    asks_institutions = bool(INSTITUTION_QUESTION_RE.search(query))
    fused = [
        (c, sc)
        for c, sc in fused
        if not (
            (status_of(c) and status_of(c)[0] == "repealed" and c.celex not in named_here)
            or (is_institution_only(c) and not asks_institutions)
        )
    ]
    if restrict_celex:
        # The question (or the user's "Law" choice) names one instrument: search only that
        # one, so look-alike provisions of other regimes (GDPR vs EUDPR vs LED) cannot take
        # its place. Court judgments are kept (they interpret the named statute and the user
        # may be asking about them); a restriction to a judgment itself keeps only that judgment.
        # Too few hits (an instrument that does not cover the topic) -> unrestricted.
        keep_case_law = "CJ" not in restrict_celex
        only = []
        case_law = 0
        for c, sc in fused:
            if c.celex == restrict_celex:
                only.append((c, sc))
            elif keep_case_law and c.unit_type == "judgment_paragraph" and case_law < MAX_CASE_LAW_WHEN_RESTRICTED:
                # Unlimited case law crowded the statute out again (judgments quote it at
                # length): 21/25 right provisions retrieved instead of 25/25. Two paragraphs
                # still surface the relevant judgment for a question about case law.
                only.append((c, sc))
                case_law += 1
        if len(only) >= 3:
            fused = only
    fused = _cap_recitals(fused, len(fused), k_candidates // 2, chunk_of=lambda x: x[0])
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
        reranked = rerank(query, fused_chunks, len(fused_chunks))
        reranked = _cap_recitals(reranked, k_final, MAX_RECITALS_IN_RESULT, chunk_of=lambda x: x[0])
        chunks = [c for c, _ in reranked]
        scores = {c.id: s for c, s in reranked}
        if pinned_ids:
            chunks, scores = _ensure_pinned(chunks, scores, fused_chunks, pinned_ids)
        chunks = expand_articles(chunks, bm25_index.chunks, n_expand, MAX_SOURCES, EXPAND_SAME_INSTRUMENT, EXPAND_BY_HITS)
        # Rulings come last, with their own small allowance, so they never use up the slots the
        # article siblings need (measured: rulings first cost two statute questions 0.2-0.25).
        chunks = expand_rulings(chunks, bm25_index.chunks, EXPAND_RULINGS if n_expand > 0 else 0, len(chunks) + MAX_RULING_CHUNKS)
        scores = {c.id: scores.get(c.id, 0.0) for c in chunks}
        return RetrievalResult(chunks=chunks, scores=scores, reranked=True)
    except RerankerUnavailable as e:
        if use_reranker:
            warnings.warn(f"reranker unavailable, using fused ranking instead: {e}")
        top = _cap_recitals(fused, k_final, MAX_RECITALS_IN_RESULT, chunk_of=lambda x: x[0])
        chunks = [c for c, _ in top]
        scores = {c.id: s for c, s in top}
        if pinned_ids:
            chunks, scores = _ensure_pinned(chunks, scores, fused_chunks, pinned_ids)
        chunks = expand_articles(chunks, bm25_index.chunks, n_expand, MAX_SOURCES, EXPAND_SAME_INSTRUMENT, EXPAND_BY_HITS)
        # Rulings come last, with their own small allowance, so they never use up the slots the
        # article siblings need (measured: rulings first cost two statute questions 0.2-0.25).
        chunks = expand_rulings(chunks, bm25_index.chunks, EXPAND_RULINGS if n_expand > 0 else 0, len(chunks) + MAX_RULING_CHUNKS)
        scores = {c.id: scores.get(c.id, 0.0) for c in chunks}
        return RetrievalResult(chunks=chunks, scores=scores, reranked=False)
