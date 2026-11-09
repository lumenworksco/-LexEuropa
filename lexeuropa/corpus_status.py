"""What the corpus holds is the ORIGINAL text of each act, not consolidated versions, so some
provisions have since been repealed or amended. This registry records the known cases so that
retrieval can avoid repealed law and answers can warn about amended law.

Verified against the acts themselves: Directive 95/46/EC was repealed with effect from
25 May 2018 by the GDPR (Article 94(1)); Articles 12 to 15 of Directive 2000/31/EC were deleted
by the Digital Services Act (Article 89(1)); Article 5(3) of Directive 2002/58/EC was replaced by
Directive 2009/136/EC to require the user's consent. Add an entry whenever one is found.
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
    "32000L0031": (
        "repealed",
        "Articles 12 to 15 of the e-Commerce Directive were deleted by the Digital Services Act (Regulation (EU) 2022/2065, Article 89); the DSA now contains the intermediary-liability rules.",
        frozenset({"12", "13", "14", "15"}),
    ),
    "32002L0058": (
        "amended",
        "This is the 2002 wording. Article 5(3) was later replaced by Directive 2009/136/EC, which requires the user's consent before information is stored on, or read from, their device.",
        frozenset({"5"}),
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
