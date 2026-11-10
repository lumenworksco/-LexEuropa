"""Out-of-date provisions, institution-only articles, and the query rewrite."""
from lexeuropa.corpus_status import is_institution_only, status_of
from lexeuropa.generation.client import GenerationError
from lexeuropa.retrieval.bm25_index import BM25Index
from lexeuropa.retrieval.reference_resolver import ReferenceResolver
from lexeuropa.retrieval.rewrite import rewrite_query
from lexeuropa.schema import Chunk
import lexeuropa.retrieval.pipeline as pipeline_mod


def _c(celex, art, para=1, text="x", unit="article_paragraph", title=None):
    return Chunk(id=f"{celex}:{unit}:{art}:{para}", celex=celex, lang="en", instrument_title="t", citation=f"Article {art}",
                 unit_type=unit, article=art if unit == "article_paragraph" else None, para_index=str(para), text=text,
                 source_url="", title=title)


class _NoDense:
    def search(self, q, k):
        return []


def test_status_registry_marks_repealed_and_amended_provisions():
    assert status_of(_c("31995L0046", "7"))[0] == "repealed"                       # the whole directive
    assert status_of(_c("31995L0046", "0", unit="recital"))[0] == "repealed"       # including its recitals
    # ePrivacy and e-Commerce are read from their consolidated texts now, so nothing is flagged
    assert status_of(_c("32000L0031", "12")) is None and status_of(_c("32002L0058", "5")) is None
    assert status_of(_c("32016R0679", "17")) is None


def test_institution_only_articles_are_recognised_by_their_title():
    assert is_institution_only(_c("32024R1689", "100", title="Administrative fines on Union institutions, bodies, offices and agencies"))
    assert not is_institution_only(_c("32024R1689", "99", title="Penalties"))
    assert not is_institution_only(_c("32024R1689", "99"))


def _retrieve(query, chunks, **kw):
    return pipeline_mod.retrieve(query, bm25_index=BM25Index(chunks), dense_index=_NoDense(), resolver=ReferenceResolver(chunks),
                                 k_final=10, use_reranker=False, expand=0, **kw)


def test_repealed_law_is_skipped_unless_the_question_names_it():
    chunks = [_c("32016R0679", "6", i, "lawful processing consent legitimate interests") for i in range(1, 5)] + \
             [_c("31995L0046", "7", i, "lawful processing consent legitimate interests consent") for i in range(1, 5)] + \
             [_c("32022R2065", "9", i, f"unrelated filler {i}") for i in range(1, 15)]
    plain = {c.celex for c in _retrieve("lawful processing consent legitimate interests", chunks).chunks}
    assert "31995L0046" not in plain and "32016R0679" in plain
    named = {c.celex for c in _retrieve("lawful processing consent under Directive 95/46", chunks).chunks}
    assert "31995L0046" in named                                                    # still reachable when asked for by name


def test_institution_only_articles_are_skipped_unless_the_question_asks():
    chunks = [_c("32024R1689", "99", 1, "administrative fines prohibited practices", title="Penalties"),
              _c("32024R1689", "99", 2, "administrative fines prohibited practices", title="Penalties"),
              _c("32024R1689", "99", 3, "administrative fines prohibited practices", title="Penalties"),
              _c("32024R1689", "100", 1, "administrative fines administrative fines prohibited practices",
                 title="Administrative fines on Union institutions, bodies, offices and agencies")] + \
             [_c("32022R2065", "9", i, f"unrelated filler {i}") for i in range(1, 15)]
    plain = [c.article for c in _retrieve("administrative fines for prohibited practices", chunks).chunks]
    assert "100" not in plain and "99" in plain
    asked = [c.article for c in _retrieve("administrative fines on Union institutions for prohibited practices", chunks).chunks]
    assert "100" in asked


def test_extra_queries_add_a_second_ranking_without_replacing_the_first():
    chunks = [_c("32016R0679", "15", 1, "right of access by the data subject confirmation purposes recipients")] + \
             [_c("32022R2065", "40", i, "data access vetted researchers access to data") for i in range(1, 5)] + \
             [_c("32022R2065", "9", i, f"unrelated filler {i}") for i in range(1, 15)]
    q = "what can a person get when they ask for access to data"
    plain = [c.article for c in _retrieve(q, chunks).chunks]
    boosted = [c.article for c in _retrieve(q, chunks, extra_queries=("right of access by the data subject confirmation purposes recipients",)).chunks]
    assert "15" in boosted
    assert "15" not in plain or boosted.index("15") < plain.index("15")     # the rewrite lifted the right article
    assert "40" in boosted                                                   # the original query still contributes its own hits


def test_rewrite_query_cleans_the_output_and_survives_failures():
    assert rewrite_query("q?", chat_fn=lambda *a, **k: ('"right of access by the data subject GDPR Article 15"\nextra line', "stop")) == \
        "right of access by the data subject GDPR Article 15"
    assert rewrite_query("q?", chat_fn=lambda *a, **k: ("ok", "stop")) is None                    # too short to be a query

    def boom(*a, **k):
        raise GenerationError("HTTP 504")

    assert rewrite_query("q?", chat_fn=boom) is None                                             # failure = no extra query


def test_expand_rulings_appends_the_ruling_of_the_top_judgments_without_reordering():
    from lexeuropa.retrieval.pipeline import expand_rulings

    def j(celex, n, op=False):
        return Chunk(id=f"{celex}:{'op' if op else 'pt'}{n}", celex=celex, lang="en", instrument_title="t", citation="c",
                     unit_type="judgment_operative" if op else "judgment_paragraph", article=None,
                     para_index=f"op{n}" if op else str(n), text="x", source_url="")

    corpus = [j("A", i) for i in (1, 2, 3)] + [j("A", 2, True), j("A", 1, True), j("B", 1, True), j("C", 1, True)]
    ranked = [j("A", 2), _c("32016R0679", "17"), j("B", 3), j("C", 5)]
    out = expand_rulings(ranked, corpus, n_cases=2, max_sources=14)
    ids = [c.id for c in out]
    assert ids[:4] == [r.id for r in ranked]                          # ranked results untouched, in order
    assert ids[4:] == ["A:op1", "A:op2", "B:op1"]                     # rulings of the first two judgments, in order
    assert "C:op1" not in ids                                         # the third is not expanded
    assert expand_rulings(ranked, corpus, n_cases=0, max_sources=14) == ranked
    assert len(expand_rulings(ranked, corpus, n_cases=2, max_sources=5)) == 5


def test_rrf_weights_let_a_hint_ranking_count_for_less_than_the_original():
    from lexeuropa.retrieval.fusion import reciprocal_rank_fusion

    a, b, c = _c("X", "1"), _c("X", "2"), _c("X", "3")
    equal = [ch.article for ch, _ in reciprocal_rank_fusion([[a, b], [c, b]])]
    weighted = [ch.article for ch, _ in reciprocal_rank_fusion([[a, b], [c, b]], weights=[1.0, 0.5])]
    assert equal[0] == "2"                           # b is in both: it wins when both count equally
    assert weighted.index("1") < weighted.index("3")  # with the second ranking down-weighted, the original's top hit beats its top hit
