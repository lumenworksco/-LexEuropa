"""Split correctness and the no-leakage guarantee for the fine-tuning dataset."""
from lexeuropa.finetune.build_dataset import split_into_claims
from lexeuropa.finetune.split import compute_split, partition_chunks
from lexeuropa.ingest.chunker import chunk_instrument
from lexeuropa.ingest.seed_corpus import SEED_INSTRUMENTS
from lexeuropa.retrieval.bm25_index import BM25Index
from lexeuropa.retrieval.reference_resolver import ReferenceResolver
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def _corpus_chunks():
    gdpr = next(i for i in SEED_INSTRUMENTS if i.celex == "32016R0679")
    case = next(i for i in SEED_INSTRUMENTS if i.celex == "62012CJ0131")
    gdpr_xhtml = (FIXTURES / "gdpr_en_excerpt.xhtml").read_text(encoding="utf-8")
    case_xhtml = (FIXTURES / "cjeu_131_12_excerpt.xhtml").read_text(encoding="utf-8")
    return chunk_instrument(gdpr, "en", gdpr_xhtml) + chunk_instrument(case, "en", case_xhtml)


def test_split_is_disjoint_and_covers_every_chunk():
    chunks = _corpus_chunks()
    split = compute_split(chunks)
    train_ids = set(split["train_ids"])
    held_ids = set(split["held_out_ids"])

    assert train_ids.isdisjoint(held_ids)
    assert train_ids | held_ids == {c.id for c in chunks}
    assert held_ids, "expected at least one held-out chunk in this fixture corpus"


def test_split_is_deterministic():
    chunks = _corpus_chunks()
    assert compute_split(chunks) == compute_split(chunks)


def test_held_out_article_17_is_never_in_the_train_split():
    # Article 17 is explicitly in GDPR_HELD_OUT_ARTICLES (used in eval/manual testing).
    chunks = _corpus_chunks()
    train, held = partition_chunks(chunks, compute_split(chunks))
    assert any(c.article == "17" for c in held)
    assert not any(c.article == "17" for c in train)


def test_bm25_and_resolver_built_from_train_split_cannot_surface_held_out_chunks():
    """Guards against held-out provisions leaking into training prompts as
    distractors — the actual leakage risk, not just the gold target."""
    chunks = _corpus_chunks()
    train, held = partition_chunks(chunks, compute_split(chunks))
    held_ids = {c.id for c in held}
    assert held_ids, "fixture must actually hold something out for this test to mean anything"

    bm25 = BM25Index(train)
    resolver = ReferenceResolver(train)

    queries = [
        "right to erasure of personal data",
        "legal basis for processing personal data",
        "consent of the data subject",
        "obligations of the controller",
    ]
    for q in queries:
        for chunk, _score in bm25.search(q, k=20):
            assert chunk.id not in held_ids
        for chunk in resolver.resolve(q, default_celex="32016R0679"):
            assert chunk.id not in held_ids


def test_split_into_claims_splits_lettered_points():
    text = (
        "1. Intro sentence. (a) first point; (b) second point; (c) third point."
    )
    claims = split_into_claims(text)
    assert len(claims) == 4
    assert claims[0] == "1. Intro sentence."
    assert claims[1].startswith("(a)")
    assert claims[3].startswith("(c)")


def test_split_into_claims_keeps_plain_paragraph_whole():
    text = "A single sentence paragraph with no lettered sub-points."
    assert split_into_claims(text) == [text]


def test_split_into_claims_ignores_inline_point_cross_references():
    text = (
        "1. The data subject shall have the right to erasure where: "
        "(a) the data are unnecessary; "
        "(b) the data subject withdraws consent according to point (a) of Article 6(1), "
        "or point (a) of Article 9(2), and where there is no other ground; "
        "(c) the data were unlawfully processed."
    )
    claims = split_into_claims(text)
    assert len(claims) == 4
    assert claims[2].startswith("(b)") and "point (a) of Article 9(2)" in claims[2]
    assert claims[3].startswith("(c)")


def _chunk(celex, unit_type, article, para, cid=None):
    from lexeuropa.schema import Chunk

    return Chunk(
        id=cid or f"{celex}:en:{unit_type}_{article}_{para}", celex=celex, lang="en",
        instrument_title="t", citation="c", unit_type=unit_type, article=article,
        para_index=para, text="x", source_url="",
    )


def test_split_rules_cover_every_unit_type_without_crashing():
    chunks = [
        _chunk("32024R1689", "recital", None, "5"),     # every 5th recital held out
        _chunk("32024R1689", "recital", None, "6"),
        _chunk("32024R1689", "annex", "Annex III", "5"),
        _chunk("32024R1689", "annex", "Annex III", "1"),
        _chunk("32024R1689", "article_paragraph", "10", "1"),  # article % 5 == 0
        _chunk("32024R1689", "article_paragraph", "11", "1"),
        _chunk("12016P/TXT", "article_paragraph", "8", "1"),
    ]
    split = compute_split(chunks)
    held = set(split["held_out_ids"])
    assert chunks[0].id in held and chunks[1].id not in held
    assert chunks[2].id in held and chunks[3].id not in held
    assert chunks[4].id in held and chunks[5].id not in held
    assert chunks[6].id not in held


def test_original_gdpr_and_judgment_split_is_frozen():
    """The 42-question held-out eval relies on these exact provisions staying
    out of training: GDPR articles 6, 7, 9, 17, 21 and every 5th article, and
    every 5th paragraph of the two original judgments."""
    chunks = [
        _chunk("32016R0679", "article_paragraph", "17", "1"),
        _chunk("32016R0679", "article_paragraph", "6", "1"),
        _chunk("32016R0679", "article_paragraph", "12", "1"),
        _chunk("62012CJ0131", "judgment_paragraph", None, "75"),
        _chunk("62012CJ0131", "judgment_paragraph", None, "76"),
    ]
    held = set(compute_split(chunks)["held_out_ids"])
    assert {chunks[0].id, chunks[1].id, chunks[3].id} <= held
    assert chunks[2].id not in held and chunks[4].id not in held


def test_unseen_eval_provisions_and_judgments_are_never_trained_on(monkeypatch):
    import lexeuropa.finetune.split as split_mod

    monkeypatch.setattr(
        split_mod, "_unseen_eval_targets", lambda: ({("32024R1689", "5")}, {"62017CJ0673"})
    )
    chunks = [
        _chunk("32024R1689", "article_paragraph", "7", "1"),   # 7 % 5 != 0, but not in eval
        _chunk("32024R1689", "article_paragraph", "5", "2"),   # eval gold article
        _chunk("62017CJ0673", "judgment_paragraph", None, "3"),  # whole judgment held out
        _chunk("62014CJ0582", "judgment_paragraph", None, "3"),  # other judgment: stride only
    ]
    held = set(split_mod.compute_split(chunks)["held_out_ids"])
    assert chunks[1].id in held and chunks[2].id in held
    assert chunks[0].id not in held and chunks[3].id not in held


def test_stratified_pool_covers_every_instrument_before_repeating():
    from lexeuropa.finetune.build_dataset import stratified_pool

    chunks = [_chunk("A", "recital", None, str(i)) for i in range(50)] + [
        _chunk("B", "recital", None, str(i)) for i in range(3)
    ]
    pool = stratified_pool(chunks, limit=4)
    assert {c.celex for c in pool[:2]} == {"A", "B"}
    assert {c.celex for c in pool[:6]} == {"A", "B"}


def test_topic_question_is_derived_from_the_heading_and_names_no_provision():
    from lexeuropa.finetune.build_dataset import build_topic_question, topic_phrase

    c = _chunk("32016R0679", "article_paragraph", "17", "1")
    c.title = "Right to erasure ('right to be forgotten')"
    assert topic_phrase(c.title) == "right to erasure ('right to be forgotten')"
    assert topic_phrase("AI literacy") == "AI literacy"          # acronyms keep their case
    q = build_topic_question(c)
    assert "right to erasure" in q and "GDPR" in q
    assert "Article" not in q and "17" not in q


def test_article_groups_collect_paragraphs_in_order_and_skip_untitled():
    from lexeuropa.finetune.build_dataset import article_groups

    a2 = _chunk("32016R0679", "article_paragraph", "17", "2"); a2.title = "T"
    a1 = _chunk("32016R0679", "article_paragraph", "17", "1"); a1.title = "T"
    untitled = _chunk("12016M/TXT", "article_paragraph", "16", "1")
    groups = article_groups([a2, a1, untitled])
    assert list(groups) == [("32016R0679", "17")]
    assert [c.para_index for c in groups[("32016R0679", "17")]] == ["1", "2"]
