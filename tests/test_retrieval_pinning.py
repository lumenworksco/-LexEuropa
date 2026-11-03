"""Regression test for a live bug: the reranker can score an exact
reference-resolver match below other text and drop it from the final
top-k entirely. Small, precise resolver hits must be pinned into the
result regardless of what the reranker (or its fallback) does with them.
"""
from pathlib import Path

from lexeuropa.ingest.chunker import chunk_instrument
from lexeuropa.ingest.seed_corpus import SEED_INSTRUMENTS
from lexeuropa.retrieval.bm25_index import BM25Index
from lexeuropa.retrieval.reference_resolver import ReferenceResolver
import lexeuropa.retrieval.pipeline as pipeline_mod
from lexeuropa.retrieval.reranker import RerankerUnavailable

FIXTURES = Path(__file__).parent / "fixtures"


def _gdpr_chunks():
    gdpr = next(i for i in SEED_INSTRUMENTS if i.celex == "32016R0679")
    xhtml = (FIXTURES / "gdpr_en_excerpt.xhtml").read_text(encoding="utf-8")
    return chunk_instrument(gdpr, "en", xhtml)


class _EmptyDenseIndex:
    """Stands in for dense retrieval without any network call."""

    def search(self, query, k):
        return []


def test_pinned_resolver_hit_survives_when_reranker_is_unavailable(monkeypatch):
    chunks = _gdpr_chunks()
    bm25 = BM25Index(chunks)
    resolver = ReferenceResolver(chunks)

    def _unavailable(*a, **k):
        raise RerankerUnavailable("forced unavailable for this test")

    monkeypatch.setattr(pipeline_mod, "rerank", _unavailable)

    result = pipeline_mod.retrieve(
        "What does Article 17(1) of the GDPR provide?",
        bm25_index=bm25,
        dense_index=_EmptyDenseIndex(),
        resolver=resolver,
        default_celex="32016R0679",
        k_final=8,
    )
    assert not result.reranked
    assert "32016R0679:en:art_17:001" in {c.id for c in result.chunks}


def test_pinned_resolver_hit_survives_a_reranker_that_drops_it(monkeypatch):
    """Reproduces the live bug directly: the reranker returns a full,
    plausible-looking top-k that simply doesn't include the resolver's
    exact match."""
    chunks = _gdpr_chunks()
    bm25 = BM25Index(chunks)
    resolver = ReferenceResolver(chunks)

    def _bad_rerank(query, candidates, top_k):
        gold_id = "32016R0679:en:art_17:001"
        decoys = [c for c in candidates if c.id != gold_id][:top_k]
        return [(c, 1.0 - i * 0.01) for i, c in enumerate(decoys)]

    monkeypatch.setattr(pipeline_mod, "rerank", _bad_rerank)

    result = pipeline_mod.retrieve(
        "What does Article 17(1) of the GDPR provide?",
        bm25_index=bm25,
        dense_index=_EmptyDenseIndex(),
        resolver=resolver,
        default_celex="32016R0679",
        k_final=8,
    )
    assert result.reranked
    assert "32016R0679:en:art_17:001" in {c.id for c in result.chunks}
    # still respects k_final, didn't just grow the list to squeeze it in
    assert len(result.chunks) == 8


def test_article_pin_survives_a_bare_case_number_in_the_same_query(monkeypatch):
    """Reproduces a live bug found while validating the resolver: a
    question naming both an exact article ("Article 17 GDPR") and a bare
    case number ("C-131/12") has a combined resolve() hit count that's
    large purely from the bare case's full-paragraph dump — gating
    pinning on that combined count (the old behavior) silently disabled
    pinning for the article hit too, and a reranker that drops it then
    loses it entirely. Pinning must be gated on resolve_precise() alone."""
    gdpr_chunks = _gdpr_chunks()
    case = next(i for i in SEED_INSTRUMENTS if i.celex == "62012CJ0131")
    case_chunks = chunk_instrument(
        case, "en", (FIXTURES / "cjeu_131_12_excerpt.xhtml").read_text()
    )
    chunks = gdpr_chunks + case_chunks
    bm25 = BM25Index(chunks)
    resolver = ReferenceResolver(chunks)

    def _bad_rerank(query, candidates, top_k):
        gold_id = "32016R0679:en:art_17:001"
        decoys = [c for c in candidates if c.id != gold_id][:top_k]
        return [(c, 1.0 - i * 0.01) for i, c in enumerate(decoys)]

    monkeypatch.setattr(pipeline_mod, "rerank", _bad_rerank)

    result = pipeline_mod.retrieve(
        "According to the GDPR, how does Article 17 relate to C-131/12?",
        bm25_index=bm25,
        dense_index=_EmptyDenseIndex(),
        resolver=resolver,
        k_final=8,
    )
    assert "32016R0679:en:art_17:001" in {c.id for c in result.chunks}


def test_bare_case_number_hits_are_not_pinned():
    """A bare case-number match can be dozens of paragraphs — too broad
    to force through — so it should participate in normal rerank/fallback
    ranking rather than being pinned wholesale."""
    gdpr = next(i for i in SEED_INSTRUMENTS if i.celex == "32016R0679")
    case = next(i for i in SEED_INSTRUMENTS if i.celex == "62012CJ0131")
    gdpr_chunks = chunk_instrument(gdpr, "en", (FIXTURES / "gdpr_en_excerpt.xhtml").read_text())
    case_chunks = chunk_instrument(
        case, "en", (FIXTURES / "cjeu_131_12_excerpt.xhtml").read_text()
    )
    chunks = gdpr_chunks + case_chunks
    resolver = ReferenceResolver(chunks)
    resolved = resolver.resolve("This was established in C-131/12.")
    assert len(resolved) > pipeline_mod.MAX_PINNED_RESOLVER_HITS
