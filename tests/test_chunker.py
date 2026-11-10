"""Structure-aware chunking, against real cached CELLAR fixtures (no network)."""
from pathlib import Path

from lexeuropa.ingest.chunker import chunk_instrument
from lexeuropa.ingest.seed_corpus import SEED_INSTRUMENTS

FIXTURES = Path(__file__).parent / "fixtures"


def _gdpr_en():
    inst = next(i for i in SEED_INSTRUMENTS if i.celex == "32016R0679")
    xhtml = (FIXTURES / "gdpr_en_excerpt.xhtml").read_text(encoding="utf-8")
    return inst, xhtml


def test_article_17_paragraph_1_survives_chunking():
    inst, xhtml = _gdpr_en()
    chunks = chunk_instrument(inst, "en", xhtml)

    art17_1 = [c for c in chunks if c.article == "17" and c.para_index == "1"]
    assert len(art17_1) == 1
    chunk = art17_1[0]

    assert chunk.celex == "32016R0679"
    assert chunk.citation == "Article 17(1) GDPR"
    assert chunk.id == "32016R0679:en:art_17:001"
    assert "erasure" in chunk.text.lower()
    assert "right to be forgotten" not in chunk.text.lower()  # that's the article title, not body


def test_multiple_paragraphs_of_an_article_are_distinct_chunks():
    inst, xhtml = _gdpr_en()
    chunks = chunk_instrument(inst, "en", xhtml)

    art17 = sorted((c for c in chunks if c.article == "17"), key=lambda c: c.para_index)
    assert len(art17) >= 2
    ids = [c.id for c in art17]
    assert len(ids) == len(set(ids))  # every paragraph gets a unique id


def test_judgment_paragraphs_get_case_citations():
    inst = next(i for i in SEED_INSTRUMENTS if i.celex == "62012CJ0131")
    xhtml = (FIXTURES / "cjeu_131_12_excerpt.xhtml").read_text(encoding="utf-8")
    chunks = chunk_instrument(inst, "en", xhtml)

    assert chunks, "expected at least one judgment paragraph chunk"
    first = next(c for c in chunks if c.para_index == "1")
    assert first.citation == "C-131/12, para. 1"
    assert first.unit_type == "judgment_paragraph"
    assert first.article is None


def test_parse_judgment_handles_newer_coj_count_markup():
    """Judgments from ~2020 on number paragraphs as <p class="coj-count">N</p>
    (no id) and use bare-dash cells for bullets. Real excerpt of C-311/18."""
    from pathlib import Path

    from lexeuropa.ingest.parser import parse_judgment

    xhtml = (Path(__file__).parent / "fixtures" / "cjeu_311_18_excerpt.xhtml").read_text()
    units = parse_judgment(xhtml)
    assert [u.para_index for u in units] == ["1", "2", "3"]
    assert all(u.unit_type == "judgment_paragraph" and len(u.text) > 20 for u in units)


def test_recitals_and_annex_are_parsed_from_real_ai_act_excerpt():
    """Real excerpt of Regulation 2024/1689: recitals 1-2 and Annex II."""
    from pathlib import Path

    from lexeuropa.ingest.chunker import chunk_instrument
    from lexeuropa.ingest.seed_corpus import SEED_INSTRUMENTS

    ai_act = next(i for i in SEED_INSTRUMENTS if i.celex == "32024R1689")
    xhtml = (Path(__file__).parent / "fixtures" / "ai_act_recital_annex_excerpt.xhtml").read_text()
    chunks = chunk_instrument(ai_act, "en", xhtml)

    recitals = [c for c in chunks if c.unit_type == "recital"]
    assert [c.id for c in recitals] == ["32024R1689:en:rct_1", "32024R1689:en:rct_2"]
    assert recitals[0].citation == "Recital 1 AI Act"
    assert "purpose of this Regulation" in recitals[0].text
    assert not recitals[0].text.startswith("(1)")

    annex = [c for c in chunks if c.unit_type == "annex"]
    assert annex and all(c.article == "Annex II" for c in annex)
    assert annex[0].id == "32024R1689:en:anx_II:001"
    assert annex[0].citation.startswith("Annex II")
    assert "ANNEX II" in annex[0].text


def test_legacy_oj_layout_charter_articles():
    """Treaties and the Charter use the older OJ layout (class="ti-art" ...).
    Real excerpt of the Charter, Articles 7 and 8."""
    from pathlib import Path

    from lexeuropa.ingest.chunker import chunk_instrument
    from lexeuropa.ingest.seed_corpus import SEED_INSTRUMENTS

    charter = next(i for i in SEED_INSTRUMENTS if i.celex == "12016P/TXT")
    xhtml = (Path(__file__).parent / "fixtures" / "charter_articles_7_8_excerpt.xhtml").read_text(
        encoding="utf-8"
    )
    chunks = chunk_instrument(charter, "en", xhtml)
    by_id = {c.id: c for c in chunks}
    assert "12016P/TXT:en:art_7:001" in by_id
    art7 = by_id["12016P/TXT:en:art_7:001"]
    assert art7.citation == "Article 7 Charter"
    assert art7.title == "Respect for private and family life"
    assert art7.text.startswith("Everyone has the right to respect")
    assert art7.index_text.startswith("Respect for private and family life. Everyone")
    assert [c.id for c in chunks if c.article == "8"] == [
        "12016P/TXT:en:art_8:001",
        "12016P/TXT:en:art_8:002",
        "12016P/TXT:en:art_8:003",
    ]
    assert by_id["12016P/TXT:en:art_8:001"].citation == "Article 8(1) Charter"


def test_plain_html_layout_directive_95_46_recitals_and_articles():
    """Pre-2010 acts come as unclassed text/html. Real excerpt of Directive 95/46/EC."""
    from pathlib import Path

    from lexeuropa.ingest.chunker import chunk_instrument
    from lexeuropa.ingest.seed_corpus import SEED_INSTRUMENTS

    inst = next(i for i in SEED_INSTRUMENTS if i.celex == "31995L0046")
    html_text = (Path(__file__).parent / "fixtures" / "directive_95_46_excerpt.html").read_text(
        encoding="utf-8"
    )
    chunks = chunk_instrument(inst, "en", html_text)
    ids = [c.id for c in chunks]
    assert ids[0] == "31995L0046:en:rct_1" and "31995L0046:en:rct_2" in ids
    assert chunks[0].citation == "Recital 1 Directive 95/46"
    art2 = next(c for c in chunks if c.article == "2")
    assert art2.title == "Definitions" and art2.text.startswith("For the purposes of this Directive")
    art1 = [c for c in chunks if c.article == "1"]
    assert [c.para_index for c in art1] == ["1", "2"]


def test_eli_article_title_is_captured_and_indexed_but_not_in_text():
    """Real GDPR excerpt: Article 17 is titled "Right to erasure ..."."""
    from pathlib import Path

    from lexeuropa.ingest.chunker import chunk_instrument
    from lexeuropa.ingest.seed_corpus import SEED_INSTRUMENTS

    gdpr = next(i for i in SEED_INSTRUMENTS if i.celex == "32016R0679")
    xhtml = (Path(__file__).parent / "fixtures" / "gdpr_en_excerpt.xhtml").read_text(encoding="utf-8")
    art17 = [c for c in chunk_instrument(gdpr, "en", xhtml) if c.article == "17"]
    assert art17 and all(c.title and c.title.startswith("Right to erasure") for c in art17)
    assert not art17[0].text.startswith("Right to erasure")        # display text stays verbatim
    assert art17[0].index_text.startswith("Right to erasure")      # indexed text carries the heading


def _consolidated_units():
    from pathlib import Path
    from lexeuropa.ingest.parser import parse_consolidated_articles

    xhtml = (Path(__file__).parent / "fixtures" / "consolidated_excerpt.xhtml").read_text(encoding="utf-8")
    return parse_consolidated_articles(xhtml)


def test_consolidated_article_reads_as_amended_and_carries_no_amendment_flags():
    # art_5 in the fixture is ePrivacy Article 5, whose paragraph 3 was replaced by Directive 2009/136/EC
    units = [u for u in _consolidated_units() if u.article == "5"]
    assert [u.para_index for u in units] == ["1", "2", "3"]
    para3 = units[2].text
    assert "has given his or her consent" in para3 and "right to refuse" not in para3
    assert not any(ch in u.text for u in units for ch in "▼►◄")


def test_consolidated_article_keeps_list_items_with_the_paragraph_they_belong_to():
    # InfoSoc Article 2 is one lead-in sentence followed by a list (a)-(e) in separate blocks
    (para,) = [u for u in _consolidated_units() if u.article == "6"]
    assert para.citation == "Article 6" and para.title == "Reproduction right"
    assert para.text.startswith("Member States shall provide for the exclusive right")
    assert "(a)" in para.text and "for authors, of their works" in para.text and "(e)" in para.text and "satellite" in para.text


def test_consolidated_article_keeps_definitions_written_as_tables():
    # e-Commerce Article 2 writes its definitions (a)-(h) as one table per definition
    (para,) = [u for u in _consolidated_units() if u.article == "7"]
    assert "‘service provider’" in para.text and "‘established service provider’" in para.text
