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
