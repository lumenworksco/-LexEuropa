"""Deterministic train/held-out split, by provision — not by example.

Splitting by individual chunk would let the fine-tune train on
near-duplicates of eval questions about the same provision, making any
before/after comparison meaningless. Holding out whole articles and a
stride of judgment paragraphs means the eval set genuinely tests
provisions the adapter never saw during training.
"""
from __future__ import annotations

import json

from lexeuropa.config import DATA_DIR
from lexeuropa.schema import Chunk

HELD_OUT_SPLIT_PATH = DATA_DIR / "finetune" / "held_out_provisions.json"

# Every 5th GDPR article (5, 10, ..., 95) for an even, non-cherry-picked
# spread, unioned with articles already used in eval/questions.jsonl or as
# manual retrieval-testing examples this session (6, 7, 9, 17, 21) — those
# must never leak into training, since they're the ones with hand-checked
# answers already on record.
GDPR_HELD_OUT_ARTICLES = sorted(set(range(5, 100, 5)) | {6, 7, 9, 17, 21})
GDPR_CELEX = "32016R0679"

# Every 5th judgment paragraph held out, same reasoning, applied to every case.
JUDGMENT_HELD_OUT_STRIDE = 5

# Every 5th recital / annex part / article of the legislation added later.
STRIDE = 5

UNSEEN_QUESTIONS_PATH = DATA_DIR.parent / "eval" / "questions_unseen.jsonl"


def _article_number(article: str) -> int:
    digits = "".join(ch for ch in article if ch.isdigit())
    return int(digits)


def _unseen_eval_targets() -> tuple[set[tuple[str, str]], set[str]]:
    """(celex, article) pairs and whole-judgment celexes named by the
    natural-language unseen-instrument eval. Those provisions must never be
    trained on, or the eval stops measuring generalization."""
    articles: set[tuple[str, str]] = set()
    judgments: set[str] = set()
    if UNSEEN_QUESTIONS_PATH.exists():
        for line in UNSEEN_QUESTIONS_PATH.read_text().splitlines():
            q = json.loads(line)
            if q.get("gold_article"):
                articles.add((q["gold_celex"], q["gold_article"]))
            else:
                judgments.add(q["gold_celex"])
    return articles, judgments


def _is_held_out(c: Chunk, unseen_articles: set, unseen_judgments: set) -> bool:
    if c.unit_type == "judgment_operative":
        return c.celex in unseen_judgments    # the ruling follows its judgment: held out only with it
    if c.unit_type == "judgment_paragraph":
        if c.celex in unseen_judgments:
            return True  # whole judgment held out for the unseen eval
        return int(c.para_index) % JUDGMENT_HELD_OUT_STRIDE == 0
    if c.unit_type == "recital":
        return int(c.para_index) % STRIDE == 0
    if c.unit_type == "annex":
        return int(c.para_index) % STRIDE == 0
    # article_paragraph
    if (c.celex, c.article) in unseen_articles:
        return True
    n = _article_number(c.article)
    if c.celex == GDPR_CELEX:
        return n in set(GDPR_HELD_OUT_ARTICLES)
    return n % STRIDE == 0


def compute_split(chunks: list[Chunk]) -> dict:
    unseen_articles, unseen_judgments = _unseen_eval_targets()
    held_ids: list[str] = []
    train_ids: list[str] = []
    for c in chunks:
        held = _is_held_out(c, unseen_articles, unseen_judgments)
        (held_ids if held else train_ids).append(c.id)
    return {"held_out_ids": held_ids, "train_ids": train_ids}


def save_split(chunks: list[Chunk]) -> dict:
    split = compute_split(chunks)
    HELD_OUT_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HELD_OUT_SPLIT_PATH.write_text(json.dumps(split, indent=2))
    return split


def load_split() -> dict:
    return json.loads(HELD_OUT_SPLIT_PATH.read_text())


def partition_chunks(
    chunks: list[Chunk], split: dict | None = None
) -> tuple[list[Chunk], list[Chunk]]:
    """Returns (train_chunks, held_out_chunks)."""
    split = split or load_split()
    held_ids = set(split["held_out_ids"])
    train = [c for c in chunks if c.id not in held_ids]
    held = [c for c in chunks if c.id in held_ids]
    return train, held


if __name__ == "__main__":
    from lexeuropa.ingest.pipeline import load_chunks

    chunks = load_chunks()
    split = save_split(chunks)
    train, held = partition_chunks(chunks, split)

    print(f"total chunks: {len(chunks)}")
    print(f"train: {len(train)}  held-out: {len(held)}  ({len(held) / len(chunks):.1%})")
    for celex in sorted({c.celex for c in chunks}):
        t = sum(1 for c in train if c.celex == celex)
        h = sum(1 for c in held if c.celex == celex)
        print(f"  {celex}: train={t} held={h}")
    print(f"written to {HELD_OUT_SPLIT_PATH}")
