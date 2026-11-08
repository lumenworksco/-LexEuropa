"""Small retrieval + citation evaluation harness.

Measures, against eval/questions.jsonl:
  - hit@k: did retrieval surface a chunk from the gold instrument/article?
  - citation precision: of the chunk ids the model actually cited, how many
    point at the gold instrument/article?
  - citation recall: did at least one cited chunk match the gold answer?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lexeuropa.answer import NotIngestedError, answer_question

DEFAULT_QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def hit_at_k(retrieved: list[dict], gold_celex: str, gold_article: str | None) -> bool:
    for c in retrieved:
        if c["celex"] != gold_celex:
            continue
        if gold_article is None or c["article"] == gold_article:
            return True
    return False


def citation_precision_recall(
    claims: list[dict],
    retrieved_by_id: dict[str, dict],
    gold_celex: str,
    gold_article: str | None,
) -> tuple[float, float]:
    cited_ids: set[str] = set()
    for claim in claims:
        cited_ids.update(claim["citations"])
    if not cited_ids:
        return 0.0, 0.0

    def matches_gold(chunk: dict) -> bool:
        return chunk["celex"] == gold_celex and (
            gold_article is None or chunk["article"] == gold_article
        )

    correct = sum(
        1 for cid in cited_ids if cid in retrieved_by_id and matches_gold(retrieved_by_id[cid])
    )
    precision = correct / len(cited_ids)
    recall = 1.0 if correct > 0 else 0.0
    return precision, recall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--out", type=Path, default=None, help="write full results JSON here")
    args = parser.parse_args()

    questions = load_questions(args.file)
    results = []

    for q in questions:
        print(f"running: {q['question'][:70]}", flush=True)
        # No default_instrument here: deriving it from the gold label would
        # let the eval's answer key configure the system it's testing.
        try:
            out = answer_question(q["question"])
        except NotIngestedError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

        retrieved = out["retrieved"]
        hit = hit_at_k(retrieved, q["gold_celex"], q["gold_article"])
        retrieved_by_id = {c["id"]: c for c in retrieved}
        precision, recall = citation_precision_recall(
            out["claims"], retrieved_by_id, q["gold_celex"], q["gold_article"]
        )
        for claim in out["claims"]:
            print(f"    [{claim['label']}] cites={claim['citations']} :: {claim['text']}", flush=True)

        results.append(
            {
                "question": q["question"],
                "hit_at_k": hit,
                "citation_precision": precision,
                "citation_recall": recall,
                "error": out.get("error"),
                "generator": out.get("generator"),
                "answers_question": out.get("answers_question"),
            }
        )
        status = "HIT" if hit else "MISS"
        print(
            f"[{status}] precision={precision:.2f} recall={recall:.2f}"
            f"{' ERROR: ' + out['error'] if out.get('error') else ''} :: {q['question'][:70]}"
        )

    n = len(results) or 1
    n_errors = sum(1 for r in results if r["error"])
    hit_rate = sum(r["hit_at_k"] for r in results) / n
    avg_precision = sum(r["citation_precision"] for r in results) / n
    avg_recall = sum(r["citation_recall"] for r in results) / n
    n_fallback = sum(1 for r in results if r.get("generator") == "fallback")
    if n_fallback:
        print(f"fallback model used for {n_fallback}/{len(results)} questions")
    print(
        f"\nhit@k: {hit_rate:.2f}  citation_precision: {avg_precision:.2f}  "
        f"citation_recall: {avg_recall:.2f}  (n={len(results)}, errors={n_errors})"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"wrote full results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
