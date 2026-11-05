from pathlib import Path

from lexeuropa.ingest.chunker import chunk_instrument
from lexeuropa.ingest.seed_corpus import SEED_INSTRUMENTS
from lexeuropa.retrieval.reference_resolver import ReferenceResolver

FIXTURES = Path(__file__).parent / "fixtures"


def _build_chunks():
    gdpr = next(i for i in SEED_INSTRUMENTS if i.celex == "32016R0679")
    google_spain = next(i for i in SEED_INSTRUMENTS if i.celex == "62012CJ0131")

    gdpr_xhtml = (FIXTURES / "gdpr_en_excerpt.xhtml").read_text(encoding="utf-8")
    case_xhtml = (FIXTURES / "cjeu_131_12_excerpt.xhtml").read_text(encoding="utf-8")

    return chunk_instrument(gdpr, "en", gdpr_xhtml) + chunk_instrument(
        google_spain, "en", case_xhtml
    )


def test_resolves_article_with_paragraph_and_explicit_alias():
    resolver = ReferenceResolver(_build_chunks())
    hits = resolver.resolve("Under Art. 17(1) GDPR, a data subject may request erasure.")
    assert len(hits) == 1
    assert hits[0].id == "32016R0679:en:art_17:001"


def test_resolves_article_without_paragraph_returns_all_paragraphs():
    resolver = ReferenceResolver(_build_chunks())
    hits = resolver.resolve("Article 17 GDPR concerns the right to erasure.")
    assert len(hits) >= 2
    assert all(h.article == "17" for h in hits)


def test_does_not_resolve_article_without_known_instrument_alias():
    resolver = ReferenceResolver(_build_chunks())
    hits = resolver.resolve("Article 17 concerns the right to erasure.")
    assert hits == []


def test_resolves_bare_case_number_to_all_its_paragraphs():
    resolver = ReferenceResolver(_build_chunks())
    hits = resolver.resolve("This was established in C-131/12.")
    assert len(hits) > 0
    assert all(h.celex == "62012CJ0131" for h in hits)


def test_unknown_case_number_resolves_to_nothing():
    resolver = ReferenceResolver(_build_chunks())
    hits = resolver.resolve("See C-999/99 for details.")
    assert hits == []


def test_resolves_article_with_comma_style_paragraph_phrasing():
    """"Article N, paragraph M" (not just "Article N(M)") must resolve —
    this exact phrasing showed up in real eval questions and silently
    fell back to unpinned semantic search before this was added."""
    resolver = ReferenceResolver(_build_chunks())
    hits = resolver.resolve("According to the GDPR, what is set out in Article 17, paragraph 1?")
    assert len(hits) == 1
    assert hits[0].id == "32016R0679:en:art_17:001"


def test_resolves_specific_judgment_paragraph_when_unambiguous():
    """A single case number plus a single "paragraph N" mention in the same
    text pins that exact judgment paragraph — previously a bare case
    number always returned every paragraph of the case, giving BM25/dense
    nothing useful to narrow down a paragraph-specific question."""
    resolver = ReferenceResolver(_build_chunks())
    hits = resolver.resolve("What did the Court state in paragraph 1 of C-131/12?")
    assert len(hits) == 1
    assert hits[0].celex == "62012CJ0131"
    assert hits[0].para_index == "1"


def test_resolves_plural_articles_joined_by_and():
    """"Articles 9 and 10" previously fell back entirely to BM25/dense —
    _ARTICLE_RE requires the literal word "Article" immediately followed
    by a digit, so "Articles" (plural) never matched anything at all."""
    resolver = ReferenceResolver(_build_chunks())
    hits = resolver.resolve("Under the GDPR, Articles 10 and 17 are both relevant here.")
    ids = {h.id for h in hits}
    assert "32016R0679:en:art_10:001" in ids
    assert "32016R0679:en:art_17:001" in ids


def test_resolves_singular_article_word_with_and_list():
    """The common lazier phrasing "Article 10 and 17" (singular word,
    plural numbers) must resolve too, not just "Articles ... and ..."."""
    resolver = ReferenceResolver(_build_chunks())
    hits = resolver.resolve("Under the GDPR, Article 10 and 17 are both relevant here.")
    ids = {h.id for h in hits}
    assert "32016R0679:en:art_10:001" in ids
    assert "32016R0679:en:art_17:001" in ids


def test_resolves_plural_articles_with_per_article_paragraphs():
    resolver = ReferenceResolver(_build_chunks())
    hits = resolver.resolve("Under the GDPR, Articles 17(1) and 10 apply.")
    ids = {h.id for h in hits}
    assert ids == {"32016R0679:en:art_17:001", "32016R0679:en:art_10:001"}


def test_ambiguous_paragraph_mention_falls_back_to_whole_case():
    """Two "paragraph N" mentions (or two cases) make the pairing
    ambiguous — fall back to the old bare-case behavior rather than guess
    which paragraph number belongs to which case."""
    resolver = ReferenceResolver(_build_chunks())
    hits = resolver.resolve(
        "Compare paragraph 1 and paragraph 2 of C-131/12 on this point."
    )
    assert len(hits) > 1
    assert all(h.celex == "62012CJ0131" for h in hits)


def test_resolve_precise_excludes_bare_case_dump():
    """A bare case mention with no single paragraph pin contributes
    nothing to resolve_precise — only resolve() carries the full dump."""
    resolver = ReferenceResolver(_build_chunks())
    precise = resolver.resolve_precise("This was established in C-131/12.")
    assert precise == []


def test_resolve_precise_includes_an_exact_article_match_alongside_a_bare_case():
    """Reproduces the live bug: a question naming both an exact article
    and a bare case number must keep the article hit in resolve_precise
    even though resolve()'s combined list (article + every paragraph of
    the case) is large — pinning must be gated on the precise count
    alone, not swamped by the bare case's dump."""
    resolver = ReferenceResolver(_build_chunks())
    text = "According to the GDPR, how does Article 17 relate to C-131/12?"
    precise = resolver.resolve_precise(text)
    assert len(precise) <= 4
    assert all(c.article == "17" for c in precise)

    full = resolver.resolve(text)
    assert len(full) > len(precise)
    assert any(c.article == "17" for c in full)
    assert any(c.celex == "62012CJ0131" for c in full)


def test_resolve_precise_includes_pinned_judgment_paragraph():
    resolver = ReferenceResolver(_build_chunks())
    precise = resolver.resolve_precise("What did the Court state in paragraph 1 of C-131/12?")
    assert len(precise) == 1
    assert precise[0].para_index == "1"


def test_short_alias_matches_whole_words_only():
    """"dsa" must not fire inside an unrelated word, and a real "DSA" mention
    resolves to the Digital Services Act when that instrument is present."""
    from lexeuropa.schema import Chunk

    dsa = Chunk(
        id="32022R2065:en:art_6:001", celex="32022R2065", lang="en",
        instrument_title="DSA", citation="Article 6(1) DSA", unit_type="article_paragraph",
        article="6", para_index="1", text="Hosting services ...", source_url="",
    )
    resolver = ReferenceResolver([dsa])
    assert [c.id for c in resolver.resolve("What does Article 6 DSA say?")] == [dsa.id]
    assert resolver.resolve("What does Article 6 of the Pseudodsatron say?") == []


def test_joined_case_secondary_number_resolves_to_same_judgment():
    from lexeuropa.config import CASE_NUMBER_ALIASES

    assert CASE_NUMBER_ALIASES["c-698/15"] == CASE_NUMBER_ALIASES["c-203/15"] == "62015CJ0203"
    assert CASE_NUMBER_ALIASES["c-594/12"] == CASE_NUMBER_ALIASES["c-293/12"]


def _recital_annex_chunks():
    from pathlib import Path

    ai_act = next(i for i in SEED_INSTRUMENTS if i.celex == "32024R1689")
    xhtml = (FIXTURES / "ai_act_recital_annex_excerpt.xhtml").read_text(encoding="utf-8")
    return chunk_instrument(ai_act, "en", xhtml)


def test_resolves_recital_and_annex_references():
    resolver = ReferenceResolver(_recital_annex_chunks())
    assert [c.id for c in resolver.resolve("What does recital 2 of the AI Act say?")] == [
        "32024R1689:en:rct_2"
    ]
    both = resolver.resolve("Compare Recitals 1 and 2 of the AI Act")
    assert {c.id for c in both} == {"32024R1689:en:rct_1", "32024R1689:en:rct_2"}
    annex = resolver.resolve("List the offences in Annex II of the AI Act")
    assert annex and all(c.unit_type == "annex" and c.article == "Annex II" for c in annex)


def test_prose_annex_without_roman_numeral_does_not_match():
    resolver = ReferenceResolver(_recital_annex_chunks())
    assert resolver.resolve("According to the AI Act, an annex is attached to the text") == []
