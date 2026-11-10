"""Provisions of the corpus that are out of date, so retrieval can avoid repealed law and answers
can warn about it.

The articles of the acts that have been amended since publication (ePrivacy, e-Commerce,
InfoSoc, AI Act) are read from the consolidated text (see `CONSOLIDATED_SOURCE` in
`ingest/seed_corpus.py`), so amended and deleted provisions already read as they do today.
What remains here is law that is repealed as a whole: Directive 95/46/EC was repealed with
effect from 25 May 2018 by the GDPR (Article 94(1)); the corpus keeps it because judgments
interpret it. Add an entry whenever one is found. Other acts may have amendments the corpus does
not yet reflect (it has no consolidated text for those, because CELLAR holds none newer).
"""
from __future__ import annotations

import re

from lexeuropa.schema import Chunk

# celex -> (kind, note, articles); articles None = the whole act (all unit types).
STATUS: dict[str, tuple[str, str, frozenset[str] | None]] = {
    "31995L0046": (
        "repealed",
        "Directive 95/46/EC was repealed with effect from 25 May 2018 by the GDPR (Regulation (EU) 2016/679, Article 94).",
        None,
    ),
}

# Provisions that apply only to EU institutions and bodies, whatever the act they sit in
# (e.g. the AI Act's fines on Union institutions, next to the fines on everyone else).
INSTITUTION_TITLE_RE = re.compile(r"Union institutions?|institutions, bodies, offices and agencies|Union bodies", re.IGNORECASE)
INSTITUTION_QUESTION_RE = re.compile(
    r"union institution|eu institution|union bod|eu bod|institutions?,? bodies|edps|european data protection supervisor|"
    r"european commission|european parliament|council of the",
    re.IGNORECASE,
)


def status_of(chunk: Chunk) -> tuple[str, str] | None:
    """("repealed" | "amended", note) for a chunk that is out of date, else None."""
    entry = STATUS.get(chunk.celex)
    if not entry:
        return None
    kind, note, articles = entry
    if articles is None:
        return kind, note
    if chunk.unit_type == "article_paragraph" and chunk.article in articles:
        return kind, note
    return None


def is_institution_only(chunk: Chunk) -> bool:
    return bool(chunk.title and INSTITUTION_TITLE_RE.search(chunk.title))
