"""Builds a larger held-out eval set: templated questions (no LLM call —
credits are unavailable) with gold labels taken directly from chunk
metadata, never from memory. Only covers provisions in the held-out split,
so it can validate a fine-tune trained on the rest of the corpus.
"""
from __future__ import annotations

import json

from lexeuropa.config import PROJECT_ROOT
from lexeuropa.finetune.split import load_split, partition_chunks
from lexeuropa.ingest.pipeline import load_chunks
from lexeuropa.schema import Chunk

OUTPUT_PATH = PROJECT_ROOT / "eval" / "questions_heldout.jsonl"

GDPR_TEMPLATES = [
    "What does Article {article}({para}) of the GDPR provide?",
    "According to the GDPR, what is set out in Article {article}, paragraph {para}?",
    "What does Article {article}({para}) of the General Data Protection Regulation state?",
]

JUDGMENT_TEMPLATES = [
    "What did the Court state in paragraph {para} of {case}?",
    "According to {case}, what did the Court hold in paragraph {para}?",
]


def build() -> list[dict]:
    chunks = load_chunks()
    _, held = partition_chunks(chunks, load_split())

    by_celex_article: dict[tuple[str, str], list[Chunk]] = {}
    by_celex: dict[str, list[Chunk]] = {}
    for c in held:
        by_celex.setdefault(c.celex, []).append(c)
        if c.unit_type == "article_paragraph":
            by_celex_article.setdefault((c.celex, c.article), []).append(c)

    questions: list[dict] = []

    # One question per held-out GDPR article, using its lowest paragraph number.
    gdpr_articles = sorted(
        {a for (celex, a) in by_celex_article if celex == "32016R0679"},
        key=lambda a: int("".join(ch for ch in a if ch.isdigit())),
    )
    for i, article in enumerate(gdpr_articles):
        para_chunks = sorted(
            by_celex_article[("32016R0679", article)], key=lambda c: int(c.para_index)
        )
        chunk = para_chunks[0]
        template = GDPR_TEMPLATES[i % len(GDPR_TEMPLATES)]
        questions.append(
            {
                "question": template.format(article=article, para=chunk.para_index),
                "lang": "en",
                "gold_celex": "32016R0679",
                "gold_article": article,
            }
        )

    # A sample of held-out judgment paragraphs per case (every other one, to
    # keep the total in range rather than using all ~36).
    for celex in ("62012CJ0131", "62017CJ0136"):
        para_chunks = sorted(
            (c for c in by_celex.get(celex, []) if c.unit_type == "judgment_paragraph"),
            key=lambda c: int(c.para_index),
        )
        case_name = para_chunks[0].citation.split(",")[0] if para_chunks else celex
        for i, chunk in enumerate(para_chunks[::2]):
            template = JUDGMENT_TEMPLATES[i % len(JUDGMENT_TEMPLATES)]
            questions.append(
                {
                    "question": template.format(para=chunk.para_index, case=case_name),
                    "lang": "en",
                    "gold_celex": celex,
                    "gold_article": None,
                }
            )

    return questions


if __name__ == "__main__":
    questions = build()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    by_celex_count: dict[str, int] = {}
    for q in questions:
        by_celex_count[q["gold_celex"]] = by_celex_count.get(q["gold_celex"], 0) + 1

    print(f"wrote {len(questions)} held-out eval questions to {OUTPUT_PATH}")
    for celex, n in by_celex_count.items():
        print(f"  {celex}: {n}")
