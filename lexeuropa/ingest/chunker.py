"""Turn a fetched, parsed document into citable Chunk objects."""
from __future__ import annotations

from lexeuropa.config import CELLAR_BASE
from lexeuropa.ingest.parser import parse
from lexeuropa.schema import Chunk, Instrument

SHORT_NAMES = {
    "32016R0679": "GDPR",
}


def chunk_instrument(instrument: Instrument, lang: str, xhtml: str) -> list[Chunk]:
    units = parse(xhtml, instrument.doc_type)
    source_url = f"{CELLAR_BASE}{instrument.celex}"
    short = SHORT_NAMES.get(instrument.celex)

    chunks: list[Chunk] = []
    for unit in units:
        if unit.unit_type == "article_paragraph":
            key = f"art_{unit.article}:{unit.para_index.zfill(3)}"
            citation = unit.citation + (f" {short}" if short else "")
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
            )
        )
    return chunks
