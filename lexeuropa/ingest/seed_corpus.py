"""The small, real seed corpus used to exercise the Stage-1 RAG pipeline.

Every entry was verified against a live CELLAR fetch before being added here
(see the ingestion session notes in CLAUDE.md) — nothing here is invented.
"""
from __future__ import annotations

from lexeuropa.schema import Instrument

SEED_INSTRUMENTS: list[Instrument] = [
    Instrument(
        celex="32016R0679",
        doc_type="regulation",
        title="Regulation (EU) 2016/679 (General Data Protection Regulation)",
        date="2016-04-27",
        languages=["en"],
        relations={},
    ),
    Instrument(
        celex="62012CJ0131",
        doc_type="judgment",
        title="Google Spain SL and Google Inc. v AEPD and Mario Costeja González",
        date="2014-05-13",
        case_number="C-131/12",
        languages=["en"],
        relations={"interprets": ["31995L0046"]},  # Directive 95/46/EC (GDPR's predecessor)
    ),
    Instrument(
        celex="62017CJ0136",
        doc_type="judgment",
        title="GC and Others v Commission nationale de l'informatique et des libertés (CNIL)",
        date="2019-09-24",
        case_number="C-136/17",
        languages=["en"],
        # GDPR Articles 9-10; the judgment's own keyword header also names
        # Directive 95/46/EC (GDPR's predecessor) alongside the Regulation.
        relations={"interprets": ["32016R0679", "31995L0046"]},
    ),
]
