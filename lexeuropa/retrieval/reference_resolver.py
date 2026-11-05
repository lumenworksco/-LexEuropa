"""Exact-match resolution of explicit legal references.

Hybrid retrieval (BM25 + dense) finds conceptually relevant text, but it is
unreliable at hitting an *exact* provision someone names directly — "Art.
6(1)(f) GDPR" or "C-131/12" should deterministically return that exact
paragraph, not whatever happens to rank highest semantically. This module
handles that case with plain regex + a small alias table, and is queried
alongside hybrid search rather than instead of it.
"""
from __future__ import annotations

import re

from lexeuropa.config import CASE_NUMBER_ALIASES, INSTRUMENT_ALIASES
from lexeuropa.schema import Chunk

_ARTICLE_RE = re.compile(
    r"\b(?:Article|Art\.?|Artikel)\s*(\d+[a-z]?)"
    r"(?:\s*(?:\(\s*(\d+)\s*\)|,\s*paragraph\s+(\d+)))?",
    re.IGNORECASE,
)
# "Articles 9 and 10", "Article 6, 7 and 9", "Articles 17(1) and 6(2)" — a
# mandatory "and" distinguishes this from a plain single "Article N" mention
# (which _ARTICLE_RE already handles on its own; the two can both fire on
# the same text harmlessly, since resolve() dedupes hits by chunk id).
_PLURAL_ARTICLES_RE = re.compile(
    r"\b(?:Articles?|Arts?\.?)\s+"
    r"(\d+[a-z]?(?:\s*\(\s*\d+\s*\))?"
    r"(?:\s*,\s*\d+[a-z]?(?:\s*\(\s*\d+\s*\))?)*"
    r"\s*,?\s*and\s+\d+[a-z]?(?:\s*\(\s*\d+\s*\))?)",
    re.IGNORECASE,
)
_ARTICLE_ITEM_RE = re.compile(r"(\d+[a-z]?)(?:\s*\(\s*(\d+)\s*\))?")
# "Recital 47", "Recitals 47 and 48", "recitals 26, 30-31" (ranges are not expanded).
_RECITAL_RE = re.compile(
    r"\bRecitals?\s+(\d+(?:\s*(?:,|and|&)\s*\d+)*)", re.IGNORECASE
)
# "Annex III": the word is case-insensitive, the roman numeral must be upper-case
# so ordinary prose like "an annex is ..." never matches.
_ANNEX_RE = re.compile(r"\b(?i:annex)\s+([IVXL]+)\b")
_CASE_RE = re.compile(r"\bC[-‑\s]?(\d{1,4})\s*/\s*(\d{2,4})\b")
# Matches a standalone "paragraph N" mention near a case number — e.g.
# "paragraph 75 of C-131/12" or "C-131/12 ... paragraph 75" — distinct from
# _ARTICLE_RE's own embedded paragraph group, which only fires right next
# to an "Article N" mention.
_STANDALONE_PARAGRAPH_RE = re.compile(r"\bparagraph\s+(\d+)\b", re.IGNORECASE)


class ReferenceResolver:
    """Looks up exact chunk hits for citations found in free text."""

    def __init__(self, chunks: list[Chunk]):
        self._by_article: dict[tuple[str, str], list[Chunk]] = {}
        self._by_article_para: dict[tuple[str, str, str], list[Chunk]] = {}
        self._by_judgment_para: dict[tuple[str, str], list[Chunk]] = {}
        self._by_recital: dict[tuple[str, str], list[Chunk]] = {}
        self._by_annex: dict[tuple[str, str], list[Chunk]] = {}
        self._by_celex: dict[str, list[Chunk]] = {}
        self._celex_present: set[str] = set()
        for c in chunks:
            self._celex_present.add(c.celex)
            self._by_celex.setdefault(c.celex, []).append(c)
            if c.unit_type == "article_paragraph" and c.article is not None:
                self._by_article.setdefault((c.celex, c.article), []).append(c)
                if c.para_index is not None:
                    self._by_article_para.setdefault(
                        (c.celex, c.article, c.para_index), []
                    ).append(c)
            elif c.unit_type == "recital" and c.para_index is not None:
                self._by_recital.setdefault((c.celex, c.para_index), []).append(c)
            elif c.unit_type == "annex" and c.article is not None:
                self._by_annex.setdefault((c.celex, c.article), []).append(c)
            elif c.unit_type == "judgment_paragraph" and c.para_index is not None:
                self._by_judgment_para.setdefault((c.celex, c.para_index), []).append(c)

    def _alias_celex_in(self, text: str) -> str | None:
        """Looks for a known instrument alias anywhere in the text — real
        questions name the instrument either right after the article
        ("Art. 17(1) GDPR") or earlier in the sentence ("According to the
        GDPR, what does Article 17 say?"); a narrow forward-only window
        missed the second, and that phrasing turned up in real eval
        questions. A whole-text scan is safe here given how small
        INSTRUMENT_ALIASES and the corpus's instrument count are — false
        positives would need an unrelated alias mention elsewhere in a
        much larger question than these."""
        lowered = text.lower()
        # Whole-word match: short aliases like "dsa" must not fire inside
        # unrelated words. Longest alias first so "regulation (eu) 2016/679"
        # is not shadowed by a shorter one.
        for alias, celex in sorted(INSTRUMENT_ALIASES.items(), key=lambda kv: -len(kv[0])):
            if celex in self._celex_present and re.search(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)", lowered
            ):
                return celex
        return None

    @staticmethod
    def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
        seen: set[str] = set()
        out: list[Chunk] = []
        for c in chunks:
            if c.id not in seen:
                seen.add(c.id)
                out.append(c)
        return out

    def resolve_precise(self, text: str, default_celex: str | None = None) -> list[Chunk]:
        """The subset of resolve()'s hits that name an exact, narrow
        target — a specific article (optionally with a paragraph), or a
        judgment paragraph pinned by an unambiguous case+paragraph
        mention. Deliberately excludes a bare case-number's full-paragraph
        dump (see resolve()) — the retrieval pipeline pins these into the
        final result, and pinning an arbitrary slice of "every paragraph
        of the case" would stand in for "the one actually meant" just as
        wrongly as not pinning at all."""
        hits: list[Chunk] = []

        for m in _PLURAL_ARTICLES_RE.finditer(text):
            celex = self._alias_celex_in(text) or default_celex
            if celex is None:
                continue
            for item in _ARTICLE_ITEM_RE.finditer(m.group(1)):
                article, para = item.group(1), item.group(2)
                if para is not None:
                    hits.extend(self._by_article_para.get((celex, article, para), []))
                else:
                    hits.extend(self._by_article.get((celex, article), []))

        for m in _ARTICLE_RE.finditer(text):
            article = m.group(1)
            para = m.group(2) or m.group(3)
            celex = self._alias_celex_in(text) or default_celex
            if celex is None:
                continue
            if para is not None:
                hits.extend(self._by_article_para.get((celex, article, para), []))
            else:
                hits.extend(self._by_article.get((celex, article), []))

        for m in _RECITAL_RE.finditer(text):
            celex = self._alias_celex_in(text) or default_celex
            if celex is None:
                continue
            for num in re.findall(r"\d+", m.group(1)):
                hits.extend(self._by_recital.get((celex, num), []))

        for m in _ANNEX_RE.finditer(text):
            celex = self._alias_celex_in(text) or default_celex
            if celex is None:
                continue
            hits.extend(self._by_annex.get((celex, f"Annex {m.group(1)}"), []))

        case_matches = list(_CASE_RE.finditer(text))
        paragraph_matches = list(_STANDALONE_PARAGRAPH_RE.finditer(text))
        if len(case_matches) == 1 and len(paragraph_matches) == 1:
            m = case_matches[0]
            key = f"c-{m.group(1)}/{m.group(2)}"
            celex = CASE_NUMBER_ALIASES.get(key)
            if celex and celex in self._celex_present:
                para_num = paragraph_matches[0].group(1)
                hits.extend(self._by_judgment_para.get((celex, para_num), []))

        return self._dedupe(hits)

    def resolve(self, text: str, default_celex: str | None = None) -> list[Chunk]:
        hits = list(self.resolve_precise(text, default_celex))
        precise_ids = {c.id for c in hits}

        case_matches = list(_CASE_RE.finditer(text))
        paragraph_matches = list(_STANDALONE_PARAGRAPH_RE.finditer(text))
        for m in case_matches:
            key = f"c-{m.group(1)}/{m.group(2)}"
            celex = CASE_NUMBER_ALIASES.get(key)
            if not celex or celex not in self._celex_present:
                continue
            # An unambiguous case+paragraph pair was already resolved
            # precisely above — don't also add the case's full dump.
            if len(case_matches) == 1 and len(paragraph_matches) == 1:
                pinned = self._by_judgment_para.get((celex, paragraph_matches[0].group(1)), [])
                if pinned and all(c.id in precise_ids for c in pinned):
                    continue
            # A bare case-number reference means "this case", not one
            # specific paragraph — surface every paragraph of it and let
            # ranking/reranking narrow it down.
            hits.extend(self._by_celex.get(celex, []))

        return self._dedupe(hits)
