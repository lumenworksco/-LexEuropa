"""Builds SFT training data without any paid HF call.

Generation and the LLM-judge are blocked by the account's depleted
credits, so training *targets* can't be model-written right now (and
doing so from memory instead would violate the same no-fabrication rule
this project applies everywhere else). Instead, targets are built
deterministically from the real chunk text: split into short claims,
each citing exactly the chunk it came from. Retrieval (BM25 + HF
embeddings + the local reranker) still works, so the *sources* placed in
each prompt are the real, realistic hybrid-retrieval output — including
real distractors — not hand-picked.

Only train-split chunks are used, and BM25/dense retrieval only ever
searches the train-split index, so held-out provisions never appear even
as distractors.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from lexeuropa.config import DENSE_INDEX_PATH, PROJECT_ROOT
from lexeuropa.finetune.split import load_split, partition_chunks
from lexeuropa.generation.prompts import build_messages, parse_response
from lexeuropa.ingest.pipeline import load_chunks
from lexeuropa.retrieval.bm25_index import BM25Index
from lexeuropa.retrieval.dense_index import DenseIndex
from lexeuropa.retrieval.pipeline import retrieve
from lexeuropa.retrieval.reference_resolver import ReferenceResolver
from lexeuropa.schema import Chunk

OUTPUT_PATH = PROJECT_ROOT / "data" / "finetune" / "sft_dataset.jsonl"
STATS_PATH = PROJECT_ROOT / "data" / "finetune" / "sft_dataset_stats.json"

# Set from --no-rerank. The CPU cross-encoder costs ~50s per example, so a
# full build takes hours; fused (BM25+dense+resolver) ranking picks the
# distractor sources in minutes instead.
USE_RERANKER = True

POINT_RE = re.compile(r"\(([a-z])\)\s+")

# {ref} is the chunk's own citation, e.g. "Article 5(1) AI Act", "Recital 47
# GDPR", "Annex III (part 2) AI Act" or "C-131/12, para. 75".
POSITIVE_TEMPLATES = [
    "What does {ref} provide?",
    "According to the text, what is set out in {ref}?",
    "What does {ref} say?",
]

# An example whose prompt + target exceed this many characters is dropped:
# at ~4 characters per token it keeps well inside --max-length (5120) so a
# long answer is never truncated mid-JSON during training.
MAX_EXAMPLE_CHARS = 16000


def stratified_pool(chunks: list[Chunk], limit: int | None) -> list[Chunk]:
    """Shuffled chunks, interleaved round-robin across instruments so a limited
    sample covers every document instead of being dominated by the longest."""
    by_celex: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_celex.setdefault(c.celex, []).append(c)
    for group in by_celex.values():
        random.shuffle(group)
    order = sorted(by_celex)
    pool: list[Chunk] = []
    depth = 0
    while any(depth < len(g) for g in by_celex.values()):
        for celex in order:
            if depth < len(by_celex[celex]):
                pool.append(by_celex[celex][depth])
        depth += 1
        if limit and len(pool) >= limit * 3:  # headroom: some get dropped as too long
            break
    return pool


def filter_dense_index(dense_index: DenseIndex, keep_ids: set[str]) -> DenseIndex:
    """A DenseIndex over a subset of another's chunks, reusing its cached
    vectors — no new embedding calls needed."""
    keep_mask = [i for i, c in enumerate(dense_index.chunks) if c.id in keep_ids]
    inst = DenseIndex.__new__(DenseIndex)
    inst.chunks = [dense_index.chunks[i] for i in keep_mask]
    inst.vectors = dense_index.vectors[keep_mask]
    return inst


def _list_markers(text: str) -> list[re.Match]:
    """Matches of POINT_RE that are real list markers: in alphabetical order
    from "(a)", each directly after ":", ";", "." (optionally followed by a
    bare "or"/"and") or the start of the text. This rejects inline
    cross-references like "point (a) of Article 6(1)", which would otherwise
    split a claim mid-sentence."""
    markers: list[re.Match] = []
    for m in POINT_RE.finditer(text):
        if m.group(1) != chr(ord("a") + len(markers)):
            continue
        before = re.sub(r"\s+(or|and)$", "", text[: m.start()].rstrip())
        if before and before[-1] not in ":;.":
            continue
        markers.append(m)
    return markers


def split_into_claims(text: str) -> list[str]:
    """One claim per lettered point ("(a) ...", "(b) ..."), or the whole
    paragraph as a single claim when there's no such structure."""
    matches = _list_markers(text)
    if len(matches) < 2:
        return [text]
    claims = []
    lead = text[: matches[0].start()].strip()
    if lead:
        claims.append(lead)
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        claims.append(f"({m.group(1)}) {text[start:end].strip()}")
    return claims


def build_question(chunk: Chunk) -> str:
    return random.choice(POSITIVE_TEMPLATES).format(ref=chunk.citation)


def build_target(gold_chunk: Chunk, gold_handle: str) -> dict:
    claims = [{"text": c, "citations": [gold_handle]} for c in split_into_claims(gold_chunk.text)]
    return {"language": "en", "claims": claims}


def build_positive_example(
    gold_chunk: Chunk, *, bm25: BM25Index, dense: DenseIndex, resolver: ReferenceResolver
) -> dict | None:
    question = build_question(gold_chunk)
    result = retrieve(
        question, bm25_index=bm25, dense_index=dense, resolver=resolver, k_candidates=20, k_final=7,
        use_reranker=USE_RERANKER,
    )
    sources = [c for c in result.chunks if c.id != gold_chunk.id]
    sources.append(gold_chunk)
    random.shuffle(sources)  # gold chunk must not always land in the same slot

    messages, handle_to_id = build_messages(question, sources)
    id_to_handle = {v: k for k, v in handle_to_id.items()}
    gold_handle = id_to_handle[gold_chunk.id]

    target = build_target(gold_chunk, gold_handle)
    target_json = json.dumps(target, ensure_ascii=False)
    if sum(len(m["content"]) for m in messages) + len(target_json) > MAX_EXAMPLE_CHARS:
        return None  # too long to train on without truncation

    # Self-check: the example must parse and its citation must exist among
    # the sources it was built from — drop it rather than trust it blindly.
    try:
        parsed = parse_response(target_json)
        cited_id = handle_to_id.get(parsed["claims"][0]["citations"][0])
        if cited_id != gold_chunk.id:
            return None
    except Exception:
        return None

    return {
        "prompt": messages,
        "completion": [{"role": "assistant", "content": target_json}],
        "meta": {
            "task": "positive",
            "gold_chunk_id": gold_chunk.id,
            "gold_handle": gold_handle,
            "num_sources": len(sources),
        },
    }


def build_refusal_example(
    off_topic_chunk: Chunk, unrelated_gold_chunk: Chunk, *, bm25: BM25Index, dense: DenseIndex,
    resolver: ReferenceResolver,
) -> dict | None:
    """A question about `off_topic_chunk`, but sources retrieved for a
    different, unrelated chunk — so the correct answer is a refusal."""
    question = build_question(off_topic_chunk)
    unrelated_question = build_question(unrelated_gold_chunk)
    result = retrieve(
        unrelated_question,
        bm25_index=bm25,
        dense_index=dense,
        resolver=resolver,
        k_candidates=20,
        k_final=7,
        use_reranker=USE_RERANKER,
    )
    # Guard against topical overlap pulling the real answer in anyway.
    sources = [c for c in result.chunks if c.article != off_topic_chunk.article or c.celex != off_topic_chunk.celex]
    if not sources or off_topic_chunk.id in {c.id for c in sources}:
        return None

    messages, _handle_to_id = build_messages(question, sources)
    target = {
        "language": "en",
        "claims": [
            {
                "text": "The provided sources do not contain information addressing this question.",
                "citations": [],
            }
        ],
    }
    target_json = json.dumps(target, ensure_ascii=False)
    try:
        parse_response(target_json)
    except Exception:
        return None

    return {
        "prompt": messages,
        "completion": [{"role": "assistant", "content": target_json}],
        "meta": {"task": "refusal", "num_sources": len(sources)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap positive examples, for a smoke test")
    parser.add_argument("--refusal-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-rerank", action="store_true", help="pick distractors from the fused ranking, skipping the slow CPU reranker")
    args = parser.parse_args()
    global USE_RERANKER
    USE_RERANKER = not args.no_rerank

    random.seed(args.seed)

    all_chunks = load_chunks()
    split = load_split()
    train_chunks, _held = partition_chunks(all_chunks, split)
    train_ids = {c.id for c in train_chunks}

    print(f"train-split chunks: {len(train_chunks)}")
    bm25 = BM25Index(train_chunks)

    full_dense = DenseIndex.load(DENSE_INDEX_PATH, {c.id: c for c in all_chunks})
    dense = filter_dense_index(full_dense, train_ids)
    resolver = ReferenceResolver(train_chunks)

    pool = stratified_pool(train_chunks, args.limit)

    # Written incrementally (flushed after every kept example), and each
    # example is wrapped in its own try/except: a multi-hour run making
    # hundreds of network calls WILL hit a transient failure eventually
    # (a timeout under memory pressure took down an earlier run entirely,
    # losing everything since output was only written at the very end).
    # One bad example should cost one example, not the whole run.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    kept_positive = 0
    failed_positive = 0
    kept_refusal = 0
    failed_refusal = 0
    gold_positions: dict[str, int] = {}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(pool):
            if args.limit and kept_positive >= args.limit:
                break
            try:
                ex = build_positive_example(chunk, bm25=bm25, dense=dense, resolver=resolver)
            except Exception as e:
                failed_positive += 1
                print(f"  warning: positive example for {chunk.id} failed: {e}", flush=True)
                ex = None
            if ex:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                f.flush()
                kept_positive += 1
                gold_handle = ex["meta"]["gold_handle"]
                gold_positions[gold_handle] = gold_positions.get(gold_handle, 0) + 1
            if (i + 1) % 25 == 0:
                print(
                    f"  positive: {i + 1}/{len(pool)} processed, {kept_positive} kept, "
                    f"{failed_positive} failed",
                    flush=True,
                )

        n_refusal_target = int(kept_positive * args.refusal_fraction)
        max_attempts = n_refusal_target * 3  # oversample; some get dropped by the overlap guard
        attempts = 0
        while kept_refusal < n_refusal_target and attempts < max_attempts:
            attempts += 1
            a, b = random.sample(train_chunks, 2)
            try:
                ex = build_refusal_example(a, b, bm25=bm25, dense=dense, resolver=resolver)
            except Exception as e:
                failed_refusal += 1
                print(f"  warning: refusal example failed: {e}", flush=True)
                ex = None
            if ex:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                f.flush()
                kept_refusal += 1
            if attempts % 25 == 0:
                print(
                    f"  refusal: attempt {attempts}/{max_attempts}, {kept_refusal}/{n_refusal_target} kept",
                    flush=True,
                )

    total = kept_positive + kept_refusal
    stats = {
        "total": total,
        "positive": kept_positive,
        "refusal": kept_refusal,
        "refusal_fraction": kept_refusal / total if total else 0,
        "failed_positive": failed_positive,
        "failed_refusal": failed_refusal,
        "gold_handle_position_histogram": dict(sorted(gold_positions.items())),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))
    print(f"wrote {total} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
