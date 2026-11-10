"""Turn a fetched, parsed document into citable Chunk objects."""
from __future__ import annotations

from lexeuropa.config import CELLAR_BASE
from lexeuropa.ingest.parser import parse, parse_consolidated_articles
from lexeuropa.schema import Chunk, Instrument

SHORT_NAMES = {
    "32016R0679": "GDPR",
    "32016L0680": "LED",
    "32018R1725": "EUDPR",
    "32022R2065": "DSA",
    "32022R1925": "DMA",
    "32022R0868": "Data Governance Act",
    "32023R2854": "Data Act",
    "32022L2555": "NIS 2",
    "32024R1689": "AI Act",
    "12016M/TXT": "TEU",
    "12016E/TXT": "TFEU",
    "12016P/TXT": "Charter",
    "31995L0046": "Directive 95/46",
    "32002L0058": "ePrivacy Directive",
    "32000L0031": "e-Commerce Directive",
    "32001L0029": "InfoSoc Directive",
    "32011L0083": "Consumer Rights Directive",
    "32005L0029": "UCPD",
    "31993L0013": "Unfair Terms Directive",
    "32019L0770": "Digital Content Directive",
    "32019L0771": "Sale of Goods Directive",
    "32024L2853": "Product Liability Directive",
    "32008R0593": "Rome I",
    "32007R0864": "Rome II",
    "32012R1215": "Brussels I Recast",
    "32003R0001": "Regulation 1/2003",
    "32004R0139": "EU Merger Regulation",
    "32003L0088": "Working Time Directive",
    "32023L0970": "Pay Transparency Directive",
    "32024L2831": "Platform Work Directive",
    "32001L0023": "Transfers of Undertakings Directive",
    "32000L0078": "Employment Equality Directive",
    "32000L0043": "Racial Equality Directive",
    "32019L1937": "Whistleblower Directive",
    "32016L0943": "Trade Secrets Directive",
    "32004L0038": "Citizens' Rights Directive",
    "32024R1624": "AML Regulation",
    "32023R1114": "MiCA",
    "32019L0790": "DSM Copyright Directive",
    "32024R2847": "Cyber Resilience Act",
    "32013R0604": "Dublin III",
    "32019L1152": "Working Conditions Directive",
    "32014L0104": "Antitrust Damages Directive",
}


def chunk_instrument(
    instrument: Instrument, lang: str, xhtml: str, consolidated_xhtml: str | None = None
) -> list[Chunk]:
    """`consolidated_xhtml`, when given, supplies the ARTICLES (as amended); the recitals and
    annexes still come from the original text, because a consolidated text has no recitals."""
    units = parse(xhtml, instrument.doc_type)
    if consolidated_xhtml is not None:
        units = [u for u in units if u.unit_type != "article_paragraph"] + parse_consolidated_articles(consolidated_xhtml)
    source_url = f"{CELLAR_BASE}{instrument.celex}"
    short = SHORT_NAMES.get(instrument.celex)

    chunks: list[Chunk] = []
    for unit in units:
        if unit.unit_type == "article_paragraph":
            key = f"art_{unit.article}:{unit.para_index.zfill(3)}"
            citation = unit.citation + (f" {short}" if short else "")
        elif unit.unit_type == "recital":
            key = f"rct_{unit.para_index}"
            citation = unit.citation + (f" {short}" if short else "")
        elif unit.unit_type == "annex":
            slug = unit.article.replace("Annex", "").strip() or "0"
            key = f"anx_{slug}:{unit.para_index.zfill(3)}"
            citation = unit.citation + (f" {short}" if short else "")
        elif unit.unit_type == "judgment_operative":
            key = f"op_{unit.para_index[2:]}"
            case = instrument.case_number or instrument.celex
            citation = f"{case}, {unit.citation}"
        else:  # judgment_paragraph
            key = f"pt_{unit.para_index}"
            case = instrument.case_number or instrument.celex
            citation = f"{case}, {unit.citation}"

        chunks.append(
            Chunk(
                id=f"{instrument.celex}:{lang}:{key}",
                celex=instrument.celex,
                lang=lang,
                instrument_title=instrument.title,
                citation=citation,
                unit_type=unit.unit_type,
                article=unit.article,
                para_index=unit.para_index,
                text=unit.text,
                source_url=source_url,
                title=unit.title,
            )
        )
    return chunks
