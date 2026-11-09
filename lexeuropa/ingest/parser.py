"""Structure-aware parsing of CELLAR XHTML into citable units.

Two document shapes are handled, based on inspecting real CELLAR output:

Legislation (regulations/directives, ELI-tagged "Official Journal" HTML):
    <div class="eli-subdivision" id="art_17">
        <p class="oj-ti-art">Article 17</p>
        <div class="eli-title"><p class="oj-sti-art">Right to erasure ...</p></div>
        <div id="017.001"><p class="oj-normal">1. ...</p></div>
        <div id="017.002">...</div>
    </div>

Judgments (CJEU case law):
    <table>...<td><p class="count" id="point93">93</p></td>
              <td><p class="normal">... judgment reasoning ...</p></td>...</table>

Both shapes are confirmed against live fetches of CELEX 32016R0679 (GDPR)
and 62012CJ0131 (Google Spain), not assumed from documentation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from lxml import html as lxml_html

ARTICLE_ID_RE = re.compile(r"^art_(\d+[a-z]?)$")
PARA_ID_RE = re.compile(r"^0*(\d+)\.0*(\d+)$")
POINT_ID_RE = re.compile(r"^point(\d+)$")
RECITAL_ID_RE = re.compile(r"^rct_(\d+)$")
ANNEX_ID_RE = re.compile(r"^anx_\w+$")

# Annexes are long lists and tables; split them on row/paragraph boundaries
# into chunks of about this many characters (a single oversized row stays whole).
ANNEX_CHUNK_CHARS = 1800


@dataclass
class ParsedUnit:
    unit_type: str  # "article_paragraph" | "recital" | "annex" | "judgment_paragraph" | "judgment_operative"
    article: str | None
    para_index: str
    citation: str
    text: str
    title: str | None = None


def _clean_text(el) -> str:
    text = el.text_content()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_legislation(xhtml: str) -> list[ParsedUnit]:
    doc = lxml_html.fromstring(xhtml.encode("utf-8"))
    units: list[ParsedUnit] = parse_recitals(doc)

    for art_div in doc.xpath('//div[@class="eli-subdivision"]'):
        art_id = art_div.get("id", "")
        m = ARTICLE_ID_RE.match(art_id)
        if not m:
            continue
        article = m.group(1)
        title_el = art_div.xpath('./div[@class="eli-title"]//p')
        title = _clean_text(title_el[0]) if title_el else None

        para_divs = [
            d
            for d in art_div.xpath("./div")
            if PARA_ID_RE.match(d.get("id", ""))
        ]

        if para_divs:
            for pdiv in para_divs:
                pm = PARA_ID_RE.match(pdiv.get("id", ""))
                para_index = str(int(pm.group(2)))
                text = _clean_text(pdiv)
                if not text:
                    continue
                units.append(
                    ParsedUnit(
                        unit_type="article_paragraph",
                        article=article,
                        para_index=para_index,
                        citation=f"Article {article}({para_index})",
                        text=text,
                        title=title,
                    )
                )
        else:
            # Single-paragraph article with no explicit numbering div: take
            # all oj-normal text directly under the article, excluding the
            # title paragraphs.
            parts = [
                _clean_text(p)
                for p in art_div.xpath(
                    './/p[@class="oj-normal" or @class="oj-final"]'
                )
            ]
            text = " ".join(p for p in parts if p)
            if text:
                units.append(
                    ParsedUnit(
                        unit_type="article_paragraph",
                        article=article,
                        para_index="1",
                        citation=f"Article {article}",
                        text=text,
                        title=title,
                    )
                )
    units.extend(parse_annexes(doc))
    return units


LEGACY_ARTICLE_RE = re.compile(r"^Article\s+(\d+[a-z]?)$", re.IGNORECASE)
LEGACY_PARA_RE = re.compile(r"^(\d+)\.\s+")
LEGACY_BODY_CLASSES = {"normal", "sp-normal", "final"}


def _legacy_blocks(doc):
    """Document-order blocks of the legacy OJ layout: (kind, text) with kind
    in ti-art | sti-art | doc-ti | body. Tables (list points "(a) ...") are
    flattened row by row; paragraphs inside tables are not repeated."""
    body = doc.find(".//body")
    for el in (body if body is not None else doc).iter("p", "table"):
        if el.tag == "table":
            if el.xpath("ancestor::table"):
                continue
            for tr in el.xpath(".//tr"):
                if not tr.xpath(".//tr"):  # innermost rows only
                    t = _clean_text(tr)
                    if t:
                        yield "body", t
            continue
        if el.xpath("ancestor::table"):
            continue
        kind = el.get("class") or ""
        text = _clean_text(el)
        if not text:
            continue
        if kind in ("ti-art", "sti-art", "doc-ti"):
            yield kind, text
        elif kind in LEGACY_BODY_CLASSES:
            yield "body", text


def parse_legacy_oj(xhtml: str) -> list[ParsedUnit]:
    """Treaties, the Charter and pre-2010 directives, which CELLAR serves in
    the older OJ markup (class="ti-art", "normal", ...) without ELI divs.

    Parsing stops at the Protocols (they repeat "Article 1..." numbering).
    The article's sub-title (e.g. "Human dignity") is kept as `title`; "(ex Article ...)" former-numbering notes are dropped."""
    doc = lxml_html.fromstring(xhtml.encode("utf-8"))
    units: list[ParsedUnit] = []
    seen: set[str] = set()

    article: str | None = None
    title = ""
    paras: list[list[str]] = []  # [[index, text], ...]
    pending_title = False

    def flush():
        nonlocal article, title, paras
        if article is not None:
            for index, text in paras:
                citation = f"Article {article}({index})" if len(paras) > 1 else f"Article {article}"
                units.append(
                    ParsedUnit("article_paragraph", article, index, citation, text, title or None)
                )
        article, title, paras = None, "", []

    started = False
    for kind, text in _legacy_blocks(doc):
        if kind == "doc-ti" and started and text.upper().startswith(("PROTOCOL", "DECLARATION")):
            break
        m = LEGACY_ARTICLE_RE.match(text) if kind == "ti-art" else None
        if m:
            flush()
            if m.group(1) in seen:
                article = None  # a repeated number: ignore until the next new article
                continue
            seen.add(m.group(1))
            article, started, pending_title = m.group(1), True, True
            continue
        if article is None:
            continue
        if kind == "sti-art":
            if pending_title and not text.startswith("(ex "):
                title = text
            pending_title = False
            continue
        if kind != "body":
            continue
        pending_title = False
        pm = LEGACY_PARA_RE.match(text)
        if pm:
            index = pm.group(1)
            if paras and paras[-1][0] == index:
                paras[-1][1] += " " + text
            else:
                paras.append([index, text])
        elif paras:
            paras[-1][1] += " " + text
        else:
            paras.append(["1", text])
    flush()
    return units


PLAIN_ADOPTED_RE = re.compile(r"^HAV(?:E|ING) ADOPTED|^HAS ADOPTED", re.IGNORECASE)
PLAIN_RECITAL_RE = re.compile(r"^\((\d+)\)\s+(.*)$")
PLAIN_NUMBERED_RE = re.compile(r"^(\d+)\.\s+(.*)$")


def parse_plain_html(xhtml: str) -> list[ParsedUnit]:
    """Pre-2010 acts that CELLAR serves only as unclassed text/html (e.g.
    Directive 95/46/EC, ePrivacy 2002/58/EC): no CSS classes, so structure is
    recovered from the text — a standalone "Article N" line, an optional short
    title line, then numbered paragraphs ("1. ...") with unnumbered
    continuation lines (points, definitions) folded into the paragraph before
    them. Recitals are the "(N) Whereas ..." paragraphs before "HAVE
    ADOPTED"; parsing stops at "Done at ..."."""
    doc = lxml_html.fromstring(xhtml.encode("utf-8"))
    lines = [_clean_text(p) for p in doc.xpath("//p")]
    lines = [t for t in lines if t]

    units: list[ParsedUnit] = []
    adopted = next((i for i, t in enumerate(lines) if PLAIN_ADOPTED_RE.match(t)), None)
    if adopted is None:
        return units

    recitals: list[list[str]] = []  # [[number, text], ...]
    for t in lines[:adopted]:
        m = PLAIN_RECITAL_RE.match(t)
        if m and (not recitals and m.group(1) == "1" or recitals and int(m.group(1)) == int(recitals[-1][0]) + 1):
            recitals.append([m.group(1), m.group(2)])
        elif recitals:
            recitals[-1][1] += " " + t
    for number, text in recitals:
        units.append(ParsedUnit("recital", None, number, f"Recital {number}", text))

    article: str | None = None
    title = ""
    paras: list[list[str]] = []
    seen: set[str] = set()
    expecting_title = False

    def flush():
        nonlocal article, title, paras
        if article is not None:
            for index, text in paras:
                citation = f"Article {article}({index})" if len(paras) > 1 else f"Article {article}"
                units.append(
                    ParsedUnit("article_paragraph", article, index, citation, text, title or None)
                )
        article, title, paras = None, "", []

    for t in lines[adopted + 1 :]:
        if t.startswith("Done at"):
            break
        m = LEGACY_ARTICLE_RE.match(t)
        if m:
            flush()
            if m.group(1) in seen:
                article, expecting_title = (None, False) if False else (m.group(1), True)
                # same heading repeated (page artefact): keep collecting into it
                continue
            seen.add(m.group(1))
            article, expecting_title = m.group(1), True
            continue
        if article is None:
            continue
        if expecting_title:
            expecting_title = False
            looks_like_title = (
                len(t) < 100
                and not t.endswith((".", ":", ";"))
                and not PLAIN_NUMBERED_RE.match(t)
                and not t.startswith("(")
            )
            if looks_like_title:
                title = t
                continue
        nm = PLAIN_NUMBERED_RE.match(t)
        if nm:
            index = nm.group(1)
            if paras and paras[-1][0] == index:
                paras[-1][1] += " " + t
            else:
                paras.append([index, t])
        elif paras:
            paras[-1][1] += " " + t
        else:
            paras.append(["1", t])
    flush()
    return units


def parse_recitals(doc) -> list[ParsedUnit]:
    """<div class="eli-subdivision" id="rct_N"><table><tr><td>(N)</td><td>text</td>..."""
    units: list[ParsedUnit] = []
    for div in doc.xpath('//div[@class="eli-subdivision"]'):
        m = RECITAL_ID_RE.match(div.get("id", ""))
        if not m:
            continue
        number = m.group(1)
        tds = div.xpath(".//td")
        if len(tds) >= 2:
            text = " ".join(t for t in (_clean_text(td) for td in tds[1:]) if t)
        else:
            text = re.sub(r"^\(\d+\)\s*", "", _clean_text(div))
        if not text:
            continue
        units.append(
            ParsedUnit(
                unit_type="recital",
                article=None,
                para_index=number,
                citation=f"Recital {number}",
                text=text,
            )
        )
    return units


def _annex_label(title_paragraphs: list[str], fallback_id: str) -> str:
    first = title_paragraphs[0] if title_paragraphs else ""
    if first.upper().startswith("ANNEX"):
        return ("Annex" + first[len("ANNEX"):]).strip()  # keep roman numerals upper-case
    return f"Annex {fallback_id}"


def parse_annexes(doc) -> list[ParsedUnit]:
    """Top-level <div id="anx_X"> blocks: title paragraphs, then paragraphs and
    tables. Packed greedily into ~ANNEX_CHUNK_CHARS chunks, never splitting a
    paragraph or table row, with the annex title repeated on every chunk so a
    chunk is self-describing."""
    units: list[ParsedUnit] = []
    for div in doc.xpath('//div[starts-with(@id, "anx_")]'):
        if not ANNEX_ID_RE.match(div.get("id", "")):
            continue
        if div.xpath('ancestor::div[starts-with(@id, "anx_")]'):
            continue  # nested sub-block, already covered by its parent

        titles = [
            _clean_text(p) for p in div.xpath('.//p[@class="oj-doc-ti"]') if _clean_text(p)
        ]
        label = _annex_label(titles, div.get("id", "")[4:])
        title_text = " ".join(titles[:2])

        pieces: list[str] = []
        for el in div.iterchildren():
            if el.tag == "p" and el.get("class") == "oj-doc-ti":
                continue
            if el.tag == "table":
                pieces.extend(
                    t for t in (_clean_text(tr) for tr in el.xpath("./tbody/tr | ./tr")) if t
                )
            else:
                t = _clean_text(el)
                if t:
                    pieces.append(t)
        if not pieces:
            continue

        groups: list[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) + 1 > ANNEX_CHUNK_CHARS:
                groups.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
        if current:
            groups.append(current)

        for k, body in enumerate(groups, start=1):
            citation = label if len(groups) == 1 else f"{label} (part {k})"
            units.append(
                ParsedUnit(
                    unit_type="annex",
                    article=label,
                    para_index=str(k),
                    citation=citation,
                    text=f"{title_text} — {body}" if title_text else body,
                )
            )
    return units


OPERATIVE_RE = re.compile(r"hereby (?:rules|declares|decides)", re.IGNORECASE)
OPERATIVE_NUMBER_RE = re.compile(r"^\d+\.$")


def parse_operative_part(doc) -> list[ParsedUnit]:
    """The ruling: after "On those grounds, the Court ... hereby rules:" come pairs of a
    number ("1.") and its text, then the signatures. These points are the holdings, the most
    quoted part of a judgment, and they are unnumbered as paragraphs, so the paragraph parser
    never sees them."""
    marks = doc.xpath("//p[contains(normalize-space(.), 'hereby ')]")
    marks = [m for m in marks if OPERATIVE_RE.search(_clean_text(m))]
    if not marks:
        return []
    blocks = [(el.get("class") or "", _clean_text(el)) for el in marks[-1].xpath("following::p")]
    units: list[ParsedUnit] = []
    i = 0
    while i + 1 < len(blocks):
        cls, text = blocks[i]
        if OPERATIVE_NUMBER_RE.match(text) and "count" in cls:
            number = text.rstrip(".")
            body = blocks[i + 1][1]
            if body:
                units.append(
                    ParsedUnit(
                        unit_type="judgment_operative",
                        article=None,
                        para_index=f"op{number}",
                        citation=f"operative part, point {number}",
                        text=body,
                        title="Ruling of the Court (operative part)",
                    )
                )
            i += 2
        elif units:
            break   # signatures, notes: the ruling is over
        else:
            i += 1
    if units:
        return units
    # No numbered points: a single unnumbered ruling follows the marker (e.g. "Directive 2006/24/EC
    # ... is invalid."), possibly over several paragraphs, ending at the signatures or the judges' names.
    body: list[str] = []
    for _cls, text in blocks:
        if not text or text.startswith(("[Signatures]", "Delivered")) or len(text.split()) <= 2:
            break
        body.append(text)
    if body:
        units.append(
            ParsedUnit("judgment_operative", None, "op1", "operative part", " ".join(body), "Ruling of the Court (operative part)")
        )
    return units


def parse_judgment(xhtml: str) -> list[ParsedUnit]:
    doc = lxml_html.fromstring(xhtml.encode("utf-8"))
    units: list[ParsedUnit] = []

    # Older judgments: <p class="count" id="pointN">. Newer ones (roughly
    # 2020 onward) use <p class="coj-count">N</p> with no id — the number is
    # the cell text, and bullet cells hold a bare dash and are skipped.
    count_ps = [
        (p, m.group(1))
        for p in doc.xpath('//p[@class="count"]')
        if (m := POINT_ID_RE.match(p.get("id", "")))
    ] or [
        (p, p.text_content().strip())
        for p in doc.xpath('//p[@class="coj-count"]')
        if p.text_content().strip().isdigit()
    ]

    for count_p, para_num in count_ps:

        row = count_p.getparent()
        while row is not None and row.tag != "tr":
            row = row.getparent()
        if row is None:
            continue

        tds = row.xpath("./td")
        if len(tds) < 2:
            continue
        text = _clean_text(tds[1])
        if not text:
            continue
        units.append(
            ParsedUnit(
                unit_type="judgment_paragraph",
                article=None,
                para_index=para_num,
                citation=f"para. {para_num}",
                text=text,
            )
        )
    units.extend(parse_operative_part(doc))
    return units


def parse(xhtml: str, doc_type: str) -> list[ParsedUnit]:
    if doc_type in ("regulation", "directive", "decision", "treaty"):
        if 'class="eli-subdivision"' in xhtml:
            return parse_legislation(xhtml)
        if 'class="ti-art"' in xhtml:
            return parse_legacy_oj(xhtml)
        return parse_plain_html(xhtml)
    if doc_type in ("judgment", "ag_opinion"):
        return parse_judgment(xhtml)
    raise ValueError(f"unknown doc_type: {doc_type!r}")
