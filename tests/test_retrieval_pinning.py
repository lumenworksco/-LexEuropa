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
        expand=0,   # this test is about pinning, not article expansion
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


def test_recital_cap_keeps_operative_articles_in_a_small_result_list():
    from lexeuropa.retrieval.pipeline import _cap_recitals
    from lexeuropa.schema import Chunk

    def c(kind, n):
        return Chunk(id=f"{kind}{n}", celex="X", lang="en", instrument_title="t", citation="c",
                     unit_type=kind, article=str(n), para_index="1", text="x", source_url="")

    ranked = [c("recital", i) for i in range(6)] + [c("article_paragraph", 9)]
    capped = _cap_recitals(ranked, 5, 2)
    assert [x.id for x in capped] == ["recital0", "recital1", "article_paragraph9", "recital2", "recital3"]
    # not enough operative text to fill k: recitals backfill rather than shrink the list
    only_recitals = [c("recital", i) for i in range(4)]
    assert len(_cap_recitals(only_recitals, 3, 1)) == 3


def test_expand_articles_appends_siblings_and_never_reorders_or_displaces_the_ranked_results():
    from lexeuropa.retrieval.pipeline import expand_articles
    from lexeuropa.schema import Chunk

    def c(celex, art, para, kind="article_paragraph"):
        return Chunk(id=f"{celex}:{art}:{para}", celex=celex, lang="en", instrument_title="t", citation="c", unit_type=kind,
                     article=art if kind == "article_paragraph" else None, para_index=str(para), text="x", source_url="")

    corpus = [c("X", "13", p) for p in (1, 2, 3, 4)] + [c("X", "5", p) for p in (1, 2)] + [c("Y", "9", p) for p in (1, 2, 3)]
    ranked = [c("X", "13", 3), c("Y", "9", 2), c("X", "5", 2), c("R", "0", 7, "recital")]
    out = expand_articles(ranked, corpus, n_articles=2, max_sources=20)
    ids = [x.id for x in out]
    assert ids[:4] == [r.id for r in ranked]                                            # ranked results untouched, in order
    assert ids[4:] == ["X:13:1", "X:13:2", "X:13:4", "Y:9:1", "Y:9:3"]                  # siblings of the top two articles, statutory order
    assert "X:5:1" not in ids                                                           # the third article is not expanded (n=2)
    assert len(ids) == len(set(ids))
    assert expand_articles(ranked, corpus, n_articles=0, max_sources=20) == ranked      # disabled = unchanged
    capped = expand_articles(ranked, corpus, n_articles=2, max_sources=6)
    assert len(capped) == 6 and [x.id for x in capped[:4]] == [r.id for r in ranked]   # the cap never cuts into the ranked results


def test_same_instrument_expansion_skips_look_alike_provisions_of_other_regimes():
    from lexeuropa.retrieval.pipeline import expand_articles
    from lexeuropa.schema import Chunk

    def c(celex, art, para):
        return Chunk(id=f"{celex}:{art}:{para}", celex=celex, lang="en", instrument_title="t", citation="c", unit_type="article_paragraph",
                     article=art, para_index=str(para), text="x", source_url="")

    corpus = [c("GDPR", "83", p) for p in (1, 2, 3)] + [c("AIACT", "99", p) for p in (1, 2)] + [c("GDPR", "58", 1), c("GDPR", "58", 2)]
    ranked = [c("GDPR", "83", 2), c("AIACT", "99", 1), c("GDPR", "58", 1)]
    mixed = {x.id for x in expand_articles(ranked, corpus, 3, 20)}
    same = {x.id for x in expand_articles(ranked, corpus, 3, 20, same_instrument=True)}
    assert "AIACT:99:2" in mixed                         # default: every top article is expanded
    assert "AIACT:99:2" not in same                      # same-instrument: the other regime's article is left alone
    assert {"GDPR:83:1", "GDPR:83:3", "GDPR:58:2"} <= same   # ...while the dominant instrument's articles are expanded
    assert all(x.id in same for x in ranked)             # and the ranked results themselves are untouched


def _chunk(celex, art, para, text):
    from lexeuropa.schema import Chunk

    return Chunk(id=f"{celex}:en:art_{art}:{para:03d}", celex=celex, lang="en", instrument_title="t", citation=f"Article {art}({para})",
                 unit_type="article_paragraph", article=art, para_index=str(para), text=text, source_url="")


def test_restrict_celex_searches_only_the_named_instrument():
    gdpr = [_chunk("32016R0679", "33", i, f"breach notification supervisory authority {i}") for i in range(1, 5)]
    eudpr = [_chunk("32018R1725", "34", i, "breach breach notification notification personal data breach notification EDPS") for i in range(1, 5)]
    chunks = gdpr + eudpr
    bm25 = BM25Index(chunks)
    resolver = ReferenceResolver(chunks)
    kwargs = dict(bm25_index=bm25, dense_index=_EmptyDenseIndex(), resolver=resolver, k_final=8, use_reranker=False, expand=0)

    free = pipeline_mod.retrieve("when must a breach notification be made", **kwargs)
    assert {c.celex for c in free.chunks} == {"32016R0679", "32018R1725"}          # unrestricted: the regimes compete
    only = pipeline_mod.retrieve("when must a breach notification be made", restrict_celex="32016R0679", **kwargs)
    assert {c.celex for c in only.chunks} == {"32016R0679"}                        # restricted: only the named one


def test_restrict_celex_falls_back_when_the_instrument_has_too_few_hits():
    gdpr = [_chunk("32016R0679", "33", 1, "something unrelated")]
    eudpr = [_chunk("32018R1725", "34", i, "breach notification EDPS") for i in range(1, 6)]
    filler = [_chunk("32022R2065", "9", i, f"unrelated filler text number {i}") for i in range(1, 15)]   # keeps BM25 weights positive
    chunks = gdpr + eudpr + filler
    kwargs = dict(bm25_index=BM25Index(chunks), dense_index=_EmptyDenseIndex(), resolver=ReferenceResolver(chunks),
                  k_final=8, use_reranker=False, expand=0)
    out = pipeline_mod.retrieve("breach notification", restrict_celex="32016R0679", **kwargs)
    assert out.chunks and {c.celex for c in out.chunks} == {"32018R1725"}          # not an empty answer for an instrument that does not cover it


def test_restricting_to_a_statute_keeps_case_law_but_drops_other_legislation():
    from lexeuropa.schema import Chunk

    gdpr = [_chunk("32016R0679", "82", i, f"compensation damage infringement {i}") for i in range(1, 4)]
    eudpr = [_chunk("32018R1725", "65", i, "compensation damage infringement compensation damage") for i in range(1, 4)]
    case = [Chunk(id=f"62021CJ0300:en:pt_{i}", celex="62021CJ0300", lang="en", instrument_title="t", citation="c", unit_type="judgment_paragraph",
                  article=None, para_index=str(i), text="compensation for non-material damage under the regulation", source_url="") for i in range(1, 4)]
    filler = [_chunk("32022R2065", "9", i, f"unrelated filler text number {i}") for i in range(1, 15)]
    chunks = gdpr + eudpr + case + filler
    kwargs = dict(bm25_index=BM25Index(chunks), dense_index=_EmptyDenseIndex(), resolver=ReferenceResolver(chunks), k_final=10, use_reranker=False, expand=0)
    out = pipeline_mod.retrieve("compensation damage infringement", restrict_celex="32016R0679", **kwargs)
    celexes = {c.celex for c in out.chunks}
    assert "32016R0679" in celexes and "62021CJ0300" in celexes     # the statute and the case law about it
    assert "32018R1725" not in celexes                              # the look-alike regime is gone
    assert sum(1 for c in out.chunks if c.unit_type == "judgment_paragraph") <= 2   # case law accompanies, never crowds
    only_case = pipeline_mod.retrieve("compensation damage infringement", restrict_celex="62021CJ0300", **kwargs)
    assert {c.celex for c in only_case.chunks} == {"62021CJ0300"}   # restricting to a judgment keeps only that judgment
