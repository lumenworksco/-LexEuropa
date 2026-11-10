"""The corpus definition (seed_corpus.py, aliases, short names) stays consistent, and the
retrieval question set's gold provisions really contain what the questions are about."""
import json
from pathlib import Path

import pytest

from lexeuropa.config import CASE_NUMBER_ALIASES, INSTRUMENT_ALIASES
from lexeuropa.ingest.chunker import SHORT_NAMES
from lexeuropa.ingest.seed_corpus import CONSOLIDATED_SOURCE, SEED_INSTRUMENTS

EU_QUESTIONS = Path(__file__).parent.parent / "eval" / "questions_retrieval_eu.jsonl"


def test_celex_numbers_are_unique_and_every_act_is_nameable():
    celexes = [i.celex for i in SEED_INSTRUMENTS]
    assert len(celexes) == len(set(celexes))
    for inst in SEED_INSTRUMENTS:
        if inst.doc_type != "judgment":
            assert inst.celex in SHORT_NAMES, f"{inst.celex} has no short name for citations"
            assert inst.celex in INSTRUMENT_ALIASES.values(), f"{inst.celex} has no alias to name it by"
        else:
            assert inst.case_number and inst.case_number.lower() in CASE_NUMBER_ALIASES


def test_every_alias_points_at_a_corpus_instrument():
    known = {i.celex for i in SEED_INSTRUMENTS}
    assert set(INSTRUMENT_ALIASES.values()) <= known
    assert set(CASE_NUMBER_ALIASES.values()) <= known


def test_consolidated_sources_are_for_legislation_in_the_corpus():
    legislation = {i.celex for i in SEED_INSTRUMENTS if i.doc_type in ("regulation", "directive")}
    for original, (consolidated, date) in CONSOLIDATED_SOURCE.items():
        assert original in legislation
        assert consolidated.startswith("0") and consolidated[1:10] == original[1:10] and consolidated.endswith(date.replace("-", ""))


def test_eu_retrieval_questions_are_grounded_in_their_gold_provision():
    from lexeuropa.ingest.pipeline import load_chunks

    try:
        chunks = load_chunks()
    except FileNotFoundError:
        pytest.skip("corpus not ingested")
    known = {c.celex for c in chunks}
    for line in EU_QUESTIONS.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        if q["gold_celex"] not in known:
            pytest.skip("corpus predates the EU expansion; re-run ingest")
        text = " ".join(
            c.text for c in chunks if c.celex == q["gold_celex"] and (q["gold_article"] is None or c.article == q["gold_article"])
        ).lower()
        assert q["check"].lower() in text, f"{q['question']!r}: {q['check']!r} not in the gold provision"


def test_paragraph_numbers_sort_naturally_including_lettered_ones():
    from lexeuropa.schema import para_sort_key

    assert sorted(["10", "2", "1a", "1", "1b"], key=para_sort_key) == ["1", "1a", "1b", "2", "10"]
