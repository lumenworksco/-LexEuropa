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


@dataclass
class ParsedUnit:
    unit_type: str  # "article_paragraph" | "judgment_paragraph"
    article: str | None
    para_index: str
    citation: str
    text: str


def _clean_text(el) -> str:
    text = el.text_content()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_legislation(xhtml: str) -> list[ParsedUnit]:
    doc = lxml_html.fromstring(xhtml.encode("utf-8"))
    units: list[ParsedUnit] = []

    for art_div in doc.xpath('//div[@class="eli-subdivision"]'):
        art_id = art_div.get("id", "")
        m = ARTICLE_ID_RE.match(art_id)
        if not m:
            continue
        article = m.group(1)

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
                    )
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
    return units


def parse(xhtml: str, doc_type: str) -> list[ParsedUnit]:
    if doc_type in ("regulation", "directive", "decision", "treaty"):
        return parse_legislation(xhtml)
    if doc_type in ("judgment", "ag_opinion"):
        return parse_judgment(xhtml)
    raise ValueError(f"unknown doc_type: {doc_type!r}")
