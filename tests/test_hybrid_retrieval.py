"""BM25 and the full hybrid pipeline against the (English-only) GDPR corpus."""
from pathlib import Path

import pytest

from lexeuropa.ingest.chunker import chunk_instrument
from lexeuropa.ingest.seed_corpus import SEED_INSTRUMENTS
from lexeuropa.retrieval.bm25_index import BM25Index
from lexeuropa.retrieval.dense_index import DenseIndex, EmbeddingError
from lexeuropa.retrieval.pipeline import retrieve
from lexeuropa.retrieval.reference_resolver import ReferenceResolver

FIXTURES = Path(__file__).parent / "fixtures"


def _gdpr_chunks():
    gdpr = next(i for i in SEED_INSTRUMENTS if i.celex == "32016R0679")
    en_xhtml = (FIXTURES / "gdpr_en_excerpt.xhtml").read_text(encoding="utf-8")
    return chunk_instrument(gdpr, "en", en_xhtml)


def test_bm25_surfaces_article_17_for_an_erasure_query():
    index = BM25Index(_gdpr_chunks())
    hits = index.search("right to erasure of personal data", k=15)
    assert hits
    assert all(c.lang == "en" for c, _ in hits)
    assert any(c.article == "17" for c, _ in hits), "Article 17 should be a keyword-search candidate"


def test_full_hybrid_pipeline_ranks_article_17_first():
    """Requires HF_TOKEN and network access to HF Inference Providers; skips cleanly if unavailable."""
    chunks = _gdpr_chunks()
    bm25 = BM25Index(chunks)
    resolver = ReferenceResolver(chunks)
    try:
        dense = DenseIndex.build(chunks)
    except EmbeddingError as e:
        pytest.skip(f"HF embeddings not available: {e}")

    result = retrieve(
        "right to erasure of personal data",
        bm25_index=bm25,
        dense_index=dense,
        resolver=resolver,
        k_candidates=10,
        k_final=3,
    )
    assert result.chunks, "expected at least one hit"
    if not result.reranked:
        pytest.skip("reranker unavailable — fused (pre-rerank) order doesn't guarantee rank 1")
    assert result.chunks[0].article == "17"
