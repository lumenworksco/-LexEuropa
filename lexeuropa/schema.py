"""Core data model: legal instruments and their citable chunks."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Instrument:
    celex: str
    doc_type: str  # "regulation" | "directive" | "judgment" | "decision" | "treaty" | "ag_opinion"
    title: str
    date: str | None = None
    case_number: str | None = None  # e.g. "C-131/12", judgments only
    languages: list[str] = field(default_factory=list)
    # e.g. {"interprets": ["32016R0679"], "amends": [], "cites": ["62012CJ0131"]}
    relations: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Chunk:
    """A single citable unit of legal text (article paragraph, judgment paragraph, ...)."""

    id: str  # "{celex}:{lang}:{unit_key}" e.g. "32016R0679:en:art_17:017.001"
    celex: str
    lang: str
    instrument_title: str
    citation: str  # human-readable, e.g. "Article 17(1) GDPR" or "C-131/12, para. 93"
    unit_type: str  # "article_paragraph" | "judgment_paragraph"
    article: str | None  # "17" for legislation, None for judgments
    para_index: str | None  # "1" (legislation) or judgment paragraph number
    text: str
    source_url: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)
