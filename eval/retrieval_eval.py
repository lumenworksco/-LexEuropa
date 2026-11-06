"""Retrieval-only evaluation: no generation, no endpoint, no cost.

For each question in a gold file, runs the real `retrieve()` and reports
  - hit@k:  is a gold chunk in the final k results?
  - MRR:    reciprocal rank of the first gold chunk in the final results
  - gold-in-pool: is a gold chunk anywhere in BM25/dense top-200 (an upper bound
    for what any final-stage ranking could achieve)

Gold = chunk of the gold instrument (and article, when the question names one).

    python eval/retrieval_eval.py --file eval/questions_unseen.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lexeuropa.config import DENSE_INDEX_PATH
from lexeuropa.ingest.pipeline import load_chunks
from lexeuropa.retrieval.bm25_index import BM25Index
from lexeuropa.retrieval.dense_index import DenseIndex
from lexeuropa.retrieval.pipeline import retrieve
from lexeuropa.retrieval.reference_resolver import ReferenceResolver


def is_gold(chunk, q: dict) -> bool:
    return chunk.celex == q["gold_celex"] and (
        q.get("gold_article") is None or chunk.article == q["gold_article"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=Path(__file__).parent / "questions_unseen.jsonl")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--verbose", action="store_true", help="list the misses")
    args = parser.parse_args()

    chunks = load_chunks()
    by_id = {c.id: c for c in chunks}
    bm25 = BM25Index(chunks)
    dense = DenseIndex.load(DENSE_INDEX_PATH, by_id)
    resolver = ReferenceResolver(chunks)

    questions = [json.loads(line) for line in args.file.read_text().splitlines() if line.strip()]
    hits = pool = 0
    rr_sum = 0.0
    for q in questions:
        result = retrieve(
            q["question"], bm25_index=bm25, dense_index=dense, resolver=resolver, k_final=args.k
        )
        rank = next((i for i, c in enumerate(result.chunks, 1) if is_gold(c, q)), None)
        in_pool = any(is_gold(c, q) for c, _ in bm25.search(q["question"], 200)) or any(
            is_gold(c, q) for c, _ in dense.search(q["question"], 200)
        )
        hits += rank is not None
        pool += in_pool
        rr_sum += 1.0 / rank if rank else 0.0
        if args.verbose and rank is None:
            top = [(c.unit_type[:3], c.article or c.para_index) for c in result.chunks]
            print(f"MISS {q['gold_celex']} art={q.get('gold_article')} :: {q['question'][:70]}\n     got {top}")
    n = len(questions)
    print(f"n={n} hit@{args.k}={hits / n:.2f} MRR={rr_sum / n:.3f} gold-in-top200-pool={pool / n:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
