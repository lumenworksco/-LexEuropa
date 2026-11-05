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
